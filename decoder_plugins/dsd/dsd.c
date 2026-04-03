/*
 * MOC - music on console
 * DSD decoder plugin — DSF (.dsf) and DSDIFF (.dff) files
 *
 * Supports three playback modes (option DSDPlaybackMode):
 *
 *   "native"  — raw DSD bytes sent straight to the audio driver using
 *               SFMT_DSD_U8.  Requires an ALSA driver that exposes
 *               SND_PCM_FORMAT_DSD_U8 (e.g. a DAC with kernel DSD support).
 *               This is true "native DSD".
 *
 *   "dop"     — DSD-over-PCM (DoP, IEC 62236-3 / DSD Alliance spec).
 *               16 DSD bits are packed into a 24-bit PCM frame together
 *               with an alternating marker byte (0x05 / 0xFA).  The
 *               output format is SFMT_S32 | SFMT_LE so that a standard
 *               PCM device passes the payload through intact; a
 *               DoP-capable DAC will recognise the markers and play DSD.
 *
 *   "pcm"     — Software decimation via a FIR low-pass filter.
 *               DSD is converted to float PCM at DSD_rate/8 Hz.
 *               Works on any audio device.
 *
 * Copyright (C) 2024  MOC contributors
 * License: GPL-2+
 */

#ifdef HAVE_CONFIG_H
# include "config.h"
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <stdint.h>
#include <errno.h>
#include <assert.h>
#include <math.h>

#define DEBUG

#include "common.h"
#include "audio.h"
#include "decoder.h"
#include "server.h"
#include "log.h"
#include "files.h"
#include "options.h"

/* =========================================================================
 * Constants
 * ====================================================================== */

#define DSD_MAX_CHANNELS   6
#define DOP_MARKER_EVEN    0x05u
#define DOP_MARKER_ODD     0xFAu
#define FIR_LEN            63
#define DSD_READ_BYTES     4096

/* =========================================================================
 * File format tag
 * ====================================================================== */
typedef enum { DSD_FMT_UNKNOWN = 0, DSD_FMT_DSF, DSD_FMT_DSDIFF } dsd_fmt_t;

/* =========================================================================
 * Playback mode
 * ====================================================================== */
typedef enum {
	DSD_MODE_NATIVE,
	DSD_MODE_DOP,
	DSD_MODE_PCM
} dsd_mode_t;

/* =========================================================================
 * FIR low-pass coefficients (sinc * Blackman window, fc=0.1)
 * ====================================================================== */
static const float dsd_fir[FIR_LEN] = {
	-0.000176f, -0.000325f, -0.000390f, -0.000289f,  0.000000f,
	 0.000450f,  0.000943f,  0.001289f,  0.001284f,  0.000746f,
	-0.000374f, -0.001830f, -0.003346f, -0.004434f, -0.004582f,
	-0.003377f, -0.000604f,  0.003530f,  0.008462f,  0.013332f,
	 0.016994f,  0.018244f,  0.016094f,  0.010062f,  0.000652f,
	-0.011406f, -0.024228f, -0.035929f, -0.044100f, -0.046622f,
	-0.041969f, -0.029276f, -0.009052f,  0.017050f,  0.047073f,
	 0.078168f,  0.107456f,  0.131906f,  0.148660f,  0.155726f,
	 0.152438f,  0.139160f,  0.116835f,  0.087128f,  0.052611f,
	 0.015811f, -0.020309f, -0.052408f, -0.078004f, -0.095189f,
	-0.103017f, -0.101537f, -0.091404f, -0.073693f, -0.050498f,
	-0.024141f,  0.002419f,  0.026742f,  0.046657f,  0.060357f,
	 0.066834f,  0.065726f,  0.057380f
};

/* =========================================================================
 * Bit-reversal LUT  (DSF stores bytes LSB-first)
 * ====================================================================== */
static uint8_t bit_rev[256];
static int     bit_rev_ready = 0;

static void build_bit_rev(void)
{
	if (bit_rev_ready) return;
	for (int i = 0; i < 256; i++) {
		uint8_t b = (uint8_t)i;
		b = (uint8_t)(((b & 0xF0u) >> 4) | ((b & 0x0Fu) << 4));
		b = (uint8_t)(((b & 0xCCu) >> 2) | ((b & 0x33u) << 2));
		b = (uint8_t)(((b & 0xAAu) >> 1) | ((b & 0x55u) << 1));
		bit_rev[i] = b;
	}
	bit_rev_ready = 1;
}

/* =========================================================================
 * Decoder state
 * ====================================================================== */
struct dsd_data {
	FILE        *f;
	dsd_fmt_t    format;
	dsd_mode_t   mode;

	int          channels;
	uint32_t     dsd_rate;
	uint32_t     pcm_rate;

	long         data_offset;
	uint64_t     total_dsd_bytes;
	int          lsb_first;

	uint32_t     dsf_block_size;
	uint64_t     bytes_decoded;

	/* Format we advertise to MOC/ALSA (always a format ALSA can open) */
	long         native_fmt;
	/* Bytes per channel per native frame (1, 2, or 4) */
	int          native_Bpc;
	/* Whether decode_native must byte-swap words before output.
	 * True when user asked for u32be/u16be but the kernel driver only
	 * accepts the LE variant (snd-usb-audio always exposes DSD_U32_LE). */
	int          native_swap;

	/* FIR state (PCM mode) */
	float        fir_buf[DSD_MAX_CHANNELS][FIR_LEN];
	int          fir_pos;

	/* DoP state */
	int          dop_phase;

	int          duration;
	struct decoder_error error;
	int          ok;
};

/* =========================================================================
 * Read helpers
 * ====================================================================== */
static uint32_t read_le32(FILE *f)
{
	uint8_t b[4];
	if (fread(b, 1, 4, f) != 4) return 0;
	return (uint32_t)b[0] | ((uint32_t)b[1]<<8) |
	       ((uint32_t)b[2]<<16) | ((uint32_t)b[3]<<24);
}
static uint32_t read_be32(FILE *f)
{
	uint8_t b[4];
	if (fread(b, 1, 4, f) != 4) return 0;
	return ((uint32_t)b[0]<<24) | ((uint32_t)b[1]<<16) |
	       ((uint32_t)b[2]<<8)  |  (uint32_t)b[3];
}
static uint64_t read_le64(FILE *f)
{
	uint8_t b[8]; uint64_t v = 0;
	if (fread(b, 1, 8, f) != 8) return 0;
	for (int i = 7; i >= 0; i--) v = (v << 8) | b[i];
	return v;
}
static uint64_t read_be64(FILE *f)
{
	uint8_t b[8]; uint64_t v = 0;
	if (fread(b, 1, 8, f) != 8) return 0;
	for (int i = 0; i < 8; i++) v = (v << 8) | b[i];
	return v;
}

/* =========================================================================
 * DSF parser
 * ====================================================================== */
static int dsf_open(struct dsd_data *d, const char *file)
{
	char id[5] = {0};
	if (fread(id, 1, 4, d->f) != 4 || memcmp(id, "DSD ", 4) != 0) {
		decoder_error(&d->error, ERROR_FATAL, 0, "Not a DSF file: %s", file);
		return 0;
	}
	read_le64(d->f); read_le64(d->f); read_le64(d->f); /* sizes + meta offset */

	if (fread(id, 1, 4, d->f) != 4 || memcmp(id, "fmt ", 4) != 0) {
		decoder_error(&d->error, ERROR_FATAL, 0, "DSF: fmt chunk missing");
		return 0;
	}
	read_le64(d->f);                             /* fmt chunk size   */
	read_le32(d->f); read_le32(d->f);            /* version, fmt id  */
	read_le32(d->f);                             /* channel type     */
	uint32_t channels   = read_le32(d->f);
	uint32_t samp_freq  = read_le32(d->f);
	read_le32(d->f);                             /* bits per sample  */
	uint64_t samp_count = read_le64(d->f);
	uint32_t block_size = read_le32(d->f);
	read_le32(d->f);                             /* reserved         */

	if (channels < 1 || channels > DSD_MAX_CHANNELS) {
		decoder_error(&d->error, ERROR_FATAL, 0,
		              "DSF: unsupported channel count %u", channels);
		return 0;
	}

	if (fread(id, 1, 4, d->f) != 4 || memcmp(id, "data", 4) != 0) {
		decoder_error(&d->error, ERROR_FATAL, 0, "DSF: data chunk missing");
		return 0;
	}
	read_le64(d->f); /* data chunk size */

	d->channels        = (int)channels;
	d->dsd_rate        = samp_freq;
	d->dsf_block_size  = block_size;
	d->lsb_first       = 1;
	d->data_offset     = ftell(d->f);
	/* samp_count = DSD samples per channel; 1 sample = 1 bit */
	d->total_dsd_bytes = (samp_count / 8) * channels;
	d->format          = DSD_FMT_DSF;
	return 1;
}

/* =========================================================================
 * DSDIFF parser
 * ====================================================================== */
static int dsdiff_open(struct dsd_data *d, const char *file)
{
	char id[5] = {0};
	if (fread(id, 1, 4, d->f) != 4 || memcmp(id, "FRM8", 4) != 0) {
		decoder_error(&d->error, ERROR_FATAL, 0,
		              "Not a DSDIFF file: %s", file);
		return 0;
	}
	read_be64(d->f);
	if (fread(id, 1, 4, d->f) != 4 || memcmp(id, "DSD ", 4) != 0) {
		decoder_error(&d->error, ERROR_FATAL, 0,
		              "DSDIFF form type is not DSD");
		return 0;
	}

	uint32_t samp_freq = 0;
	uint16_t channels  = 0;
	int      found_data = 0;

	while (!feof(d->f)) {
		char ck[5] = {0};
		if (fread(ck, 1, 4, d->f) != 4) break;
		uint64_t ck_size  = read_be64(d->f);
		long     ck_start = ftell(d->f);

		if (memcmp(ck, "PROP", 4) == 0) {
			fread(id, 1, 4, d->f);
			long prop_end = ck_start + (long)ck_size;
			while (ftell(d->f) < prop_end && !feof(d->f)) {
				char sc[5] = {0};
				if (fread(sc, 1, 4, d->f) != 4) break;
				uint64_t sc_size  = read_be64(d->f);
				long     sc_start = ftell(d->f);
				if      (memcmp(sc, "FS  ", 4) == 0)
					samp_freq = read_be32(d->f);
				else if (memcmp(sc, "CHNL", 4) == 0)
					channels = (uint16_t)(read_be32(d->f) >> 16);
				fseek(d->f, sc_start + (long)sc_size, SEEK_SET);
			}
		} else if (memcmp(ck, "DST ", 4) == 0) {
			decoder_error(&d->error, ERROR_FATAL, 0,
			              "DST-compressed DSDIFF is not supported");
			return 0;
		} else if (memcmp(ck, "DSD ", 4) == 0) {
			d->data_offset     = ftell(d->f);
			d->total_dsd_bytes = ck_size;
			found_data = 1;
			break;
		}
		long next = ck_start + (long)ck_size;
		if (next & 1) next++;
		fseek(d->f, next, SEEK_SET);
	}

	if (!found_data) {
		decoder_error(&d->error, ERROR_FATAL, 0,
		              "DSDIFF: DSD data chunk not found in %s", file);
		return 0;
	}
	if (channels < 1 || channels > DSD_MAX_CHANNELS) {
		decoder_error(&d->error, ERROR_FATAL, 0,
		              "DSDIFF: unsupported channel count %u", channels);
		return 0;
	}

	d->channels        = (int)channels;
	d->dsd_rate        = samp_freq;
	d->lsb_first       = 0;
	d->format          = DSD_FMT_DSDIFF;
	return 1;
}

/* =========================================================================
 * Read interleaved DSD bytes into per-channel buffers
 * ====================================================================== */
static int dsd_read(struct dsd_data *d,
                    uint8_t buf[][DSD_READ_BYTES], int req)
{
	int ch, i;

	if (d->format == DSD_FMT_DSF) {
		uint64_t bytes_per_ch = d->bytes_decoded / d->channels;
		int blk = (int)d->dsf_block_size;
		for (ch = 0; ch < d->channels; ch++) {
			long pos = d->data_offset
			           + (long)((bytes_per_ch / blk)
			                    * ((uint64_t)blk * d->channels))
			           + (long)(ch * blk)
			           + (long)(bytes_per_ch % blk);
			fseek(d->f, pos, SEEK_SET);
			int r = (int)fread(buf[ch], 1, req, d->f);
			if (r <= 0) return 0;
			if (r < req) memset(buf[ch] + r, 0, req - r);
		}
		return req;
	} else {
		uint8_t frame[DSD_MAX_CHANNELS];
		int got = 0;
		for (i = 0; i < req; i++) {
			if (fread(frame, 1, d->channels, d->f) != (size_t)d->channels)
				break;
			for (ch = 0; ch < d->channels; ch++)
				buf[ch][i] = frame[ch];
			got++;
		}
		return got;
	}
}

/* =========================================================================
 * MODE: NATIVE — raw DSD bytes → SFMT_DSD_U8 / SFMT_DSD_U16 / SFMT_DSD_U32
 *
 * Output layout: interleaved, native_Bpc bytes per channel per frame.
 * All formats use MSB-first DSD bit ordering within each byte
 * (oldest DSD bit = MSB of byte 0 of each container word), which is
 * the ALSA DSD convention.
 *
 * For U16/U32 containers the DSD bytes fill the word MSB-first:
 *   U16: [dsd_byte_0][dsd_byte_1]  in the endianness of native_fmt
 *   U32: [dsd_byte_0]...[dsd_byte_3] in the endianness of native_fmt
 * ====================================================================== */
static int decode_native(struct dsd_data *d, char *out, int out_len)
{
	int Bpc = d->native_Bpc;               /* bytes per channel per frame */
	int req = out_len / (d->channels * Bpc);
	if (req <= 0) return 0;
	if (req > DSD_READ_BYTES / Bpc) req = DSD_READ_BYTES / Bpc;

	uint64_t bytes_per_ch_left =
		(d->total_dsd_bytes - d->bytes_decoded) / d->channels;
	if (bytes_per_ch_left == 0) return 0;
	if ((uint64_t)req > bytes_per_ch_left / Bpc)
		req = (int)(bytes_per_ch_left / Bpc);
	if (req <= 0) return 0;

	/* We read Bpc DSD bytes per channel per output frame */
	static uint8_t raw[DSD_MAX_CHANNELS][DSD_READ_BYTES];
	int got = dsd_read(d, raw, req * Bpc);
	if (got <= 0) return 0;
	/* round down to complete container words */
	got = (got / Bpc) * Bpc;
	if (got <= 0) return 0;
	int frames_got = got / Bpc;

	int want_be = (d->native_fmt & SFMT_BE) != 0;
	uint8_t *p = (uint8_t *)out;

	for (int i = 0; i < frames_got; i++) {
		for (int ch = 0; ch < d->channels; ch++) {
			/* Collect Bpc DSD source bytes, normalise to MSB-first */
			uint8_t b[4];
			for (int k = 0; k < Bpc; k++) {
				uint8_t src = raw[ch][i * Bpc + k];
				b[k] = d->lsb_first ? bit_rev[src] : src;
			}
			/* b[0..Bpc-1] now contains DSD bytes in MSB-first order.
			 * Pack into output word in the requested endianness. */
			if (Bpc == 1) {
				*p++ = b[0];
			} else if (Bpc == 2) {
				/* native_swap: user wants BE but driver is LE
				 * (or vice-versa) — swap the two bytes */
				if (d->native_swap) {
					*p++ = b[1]; *p++ = b[0];
				} else if (want_be) {
					*p++ = b[0]; *p++ = b[1];
				} else {
					*p++ = b[1]; *p++ = b[0];
				}
			} else { /* Bpc == 4 */
				if (d->native_swap) {
					/* Swap all 4 bytes */
					*p++ = b[3]; *p++ = b[2];
					*p++ = b[1]; *p++ = b[0];
				} else if (want_be) {
					*p++ = b[0]; *p++ = b[1];
					*p++ = b[2]; *p++ = b[3];
				} else {
					*p++ = b[3]; *p++ = b[2];
					*p++ = b[1]; *p++ = b[0];
				}
			}
		}
	}

	d->bytes_decoded += (uint64_t)frames_got * Bpc * d->channels;
	return frames_got * Bpc * d->channels;
}

/* =========================================================================
 * MODE: DoP  (DSD-over-PCM, IEC 62236-3)
 *
 * Output: SFMT_S32 | SFMT_LE, interleaved by channel.
 * Each 32-bit word:  [0x00][marker][DSD_hi][DSD_lo]  (LE in memory: lo,hi,marker,0x00)
 * ====================================================================== */
static int decode_dop(struct dsd_data *d, char *out, int out_len)
{
	int frames = out_len / (4 * d->channels);
	if (frames <= 0) return 0;

	uint64_t bytes_per_ch_left =
		(d->total_dsd_bytes - d->bytes_decoded) / d->channels;
	if (bytes_per_ch_left < 2) return 0;

	int req = frames * 2;
	if (req > DSD_READ_BYTES) req = DSD_READ_BYTES & ~1;
	if ((uint64_t)req > bytes_per_ch_left) req = (int)bytes_per_ch_left & ~1;
	if (req < 2) return 0;

	static uint8_t raw[DSD_MAX_CHANNELS][DSD_READ_BYTES];
	int got = dsd_read(d, raw, req);
	if (got < 2) return 0;
	got &= ~1;

	uint8_t *p = (uint8_t *)out;
	int frames_out = 0;

	for (int i = 0; i < got; i += 2) {
		uint8_t marker = d->dop_phase ? DOP_MARKER_ODD : DOP_MARKER_EVEN;
		d->dop_phase ^= 1;

		for (int ch = 0; ch < d->channels; ch++) {
			uint8_t hi = raw[ch][i];
			uint8_t lo = raw[ch][i + 1];
			if (d->lsb_first) { hi = bit_rev[hi]; lo = bit_rev[lo]; }
			/* S32_LE word in memory: [lo][hi][marker][0x00] */
			*p++ = lo;
			*p++ = hi;
			*p++ = marker;
			*p++ = 0x00u;
		}
		frames_out++;
	}

	d->bytes_decoded += (uint64_t)got * d->channels;
	return frames_out * 4 * d->channels;
}

/* =========================================================================
 * MODE: PCM  (software FIR decimation)  → SFMT_FLOAT
 * ====================================================================== */
static int decode_pcm(struct dsd_data *d, char *out, int out_len)
{
	int req = out_len / ((int)sizeof(float) * d->channels);
	if (req <= 0) return 0;
	if (req > DSD_READ_BYTES) req = DSD_READ_BYTES;

	uint64_t bytes_per_ch_left =
		(d->total_dsd_bytes - d->bytes_decoded) / d->channels;
	if (bytes_per_ch_left == 0) return 0;
	if ((uint64_t)req > bytes_per_ch_left) req = (int)bytes_per_ch_left;

	static uint8_t raw[DSD_MAX_CHANNELS][DSD_READ_BYTES];
	int got = dsd_read(d, raw, req);
	if (got <= 0) return 0;

	float *fout = (float *)out;
	int samples_out = 0;

	for (int i = 0; i < got; i++) {
		for (int ch = 0; ch < d->channels; ch++) {
			uint8_t byte = raw[ch][i];
			if (d->lsb_first) byte = bit_rev[byte];

			/* Push 8 DSD bits into circular delay line */
			for (int bit = 7; bit >= 0; bit--) {
				float s = ((byte >> bit) & 1) ? +1.0f : -1.0f;
				d->fir_buf[ch][d->fir_pos % FIR_LEN] = s;
				d->fir_pos++;
			}

			/* Convolve */
			float acc = 0.0f;
			int pos = d->fir_pos;
			for (int k = 0; k < FIR_LEN; k++)
				acc += dsd_fir[k]
				       * d->fir_buf[ch][(pos - k - 1 + FIR_LEN * 64) % FIR_LEN];
			if (acc >  1.0f) acc =  1.0f;
			if (acc < -1.0f) acc = -1.0f;
			*fout++ = acc;
		}
		samples_out++;
	}

	d->bytes_decoded += (uint64_t)got * d->channels;
	return samples_out * (int)sizeof(float) * d->channels;
}

/* =========================================================================
 * Decoder API
 * ====================================================================== */
static void *dsd_open(const char *file)
{
	struct dsd_data *d = xmalloc(sizeof(struct dsd_data));
	memset(d, 0, sizeof(*d));
	decoder_error_init(&d->error);
	d->ok = 0;

	build_bit_rev();

	d->f = fopen(file, "rb");
	if (!d->f) {
		char *err = xstrerror(errno);
		decoder_error(&d->error, ERROR_FATAL, 0, "Can't open file: %s", err);
		free(err);
		return d;
	}

	char magic[5] = {0};
	fread(magic, 1, 4, d->f);
	rewind(d->f);

	int parse_ok;
	if      (memcmp(magic, "DSD ", 4) == 0) parse_ok = dsf_open(d, file);
	else if (memcmp(magic, "FRM8", 4) == 0) parse_ok = dsdiff_open(d, file);
	else {
		decoder_error(&d->error, ERROR_FATAL, 0,
		              "Unknown DSD format: %s", file);
		return d;
	}
	if (!parse_ok) return d;

	/* Choose mode */
	const char *ms = options_get_str("DSDPlaybackMode");
	if      (ms && !strcasecmp(ms, "dop")) d->mode = DSD_MODE_DOP;
	else if (ms && !strcasecmp(ms, "pcm")) d->mode = DSD_MODE_PCM;
	else                                   d->mode = DSD_MODE_NATIVE;

	/* For native mode, read the container format the DAC supports */
	if (d->mode == DSD_MODE_NATIVE) {
		const char *nf = options_get_str("DSDNativeFormat");
		d->native_swap = 0;

		if (nf && !strcasecmp(nf, "u16le")) {
			d->native_fmt = SFMT_DSD_U16 | SFMT_LE;
			d->native_Bpc = 2;
		} else if (nf && !strcasecmp(nf, "u16be")) {
			/* Advertise LE to ALSA (kernel only exposes LE),
			 * byte-swap the 2-byte words in the decoder */
			d->native_fmt = SFMT_DSD_U16 | SFMT_LE;
			d->native_Bpc = 2;
			d->native_swap = 1;
		} else if (nf && !strcasecmp(nf, "u32le")) {
			d->native_fmt = SFMT_DSD_U32 | SFMT_LE;
			d->native_Bpc = 4;
		} else if (nf && !strcasecmp(nf, "u32be")) {
			/* Same: advertise LE, swap 4-byte words internally */
			d->native_fmt = SFMT_DSD_U32 | SFMT_LE;
			d->native_Bpc = 4;
			d->native_swap = 1;
		} else {
			/* default: u8 */
			d->native_fmt = SFMT_DSD_U8;
			d->native_Bpc = 1;
		}

		logit("DSD: native format requested=%s → advertising %s%s to ALSA",
		      nf ? nf : "u8",
		      d->native_fmt == SFMT_DSD_U8              ? "DSD_U8"    :
		      d->native_fmt == (SFMT_DSD_U16 | SFMT_LE) ? "DSD_U16_LE" :
		      d->native_fmt == (SFMT_DSD_U32 | SFMT_LE) ? "DSD_U32_LE" : "?",
		      d->native_swap ? " (with internal byte-swap)" : "");
	}

	/* Output sample-rate */
	switch (d->mode) {
	case DSD_MODE_NATIVE:
		/* ALSA counts frames: one frame = native_Bpc bytes per channel.
		 * rate = DSD_bitrate / (8 * native_Bpc) */
		d->pcm_rate = d->dsd_rate / (8 * d->native_Bpc);
		break;
	case DSD_MODE_DOP:
		/* Two DSD bytes per 32-bit PCM frame: rate = DSD_bitrate / 16 */
		d->pcm_rate = d->dsd_rate / 16;
		break;
	case DSD_MODE_PCM:
		d->pcm_rate = d->dsd_rate / 8;
		break;
	}

	if (d->dsd_rate > 0 && d->channels > 0) {
		uint64_t bytes_per_ch = d->total_dsd_bytes / d->channels;
		uint32_t bytes_per_sec_per_ch = d->dsd_rate / 8;
		d->duration = (int)(bytes_per_ch / bytes_per_sec_per_ch);
	}

	fseek(d->f, d->data_offset, SEEK_SET);

	logit("DSD: '%s' fmt=%s ch=%d dsd_rate=%u mode=%s%s pcm_rate=%u dur=%ds",
	      file,
	      d->format == DSD_FMT_DSF ? "DSF" : "DSDIFF",
	      d->channels, d->dsd_rate,
	      d->mode == DSD_MODE_NATIVE ? "native/" :
	      d->mode == DSD_MODE_DOP    ? "DoP"     : "PCM",
	      d->mode == DSD_MODE_NATIVE ?
	          (d->native_fmt == SFMT_DSD_U8              ? "u8"     :
	           d->native_fmt == (SFMT_DSD_U32 | SFMT_LE) ?
	               (d->native_swap                       ? "u32be→u32le(swap)"
	                                                     : "u32le") :
	           d->native_fmt == (SFMT_DSD_U16 | SFMT_LE) ?
	               (d->native_swap                       ? "u16be→u16le(swap)"
	                                                     : "u16le") : "?")
	          : "",
	      d->pcm_rate, d->duration);

	d->ok = 1;
	return d;
}

static void dsd_close(void *prv)
{
	struct dsd_data *d = (struct dsd_data *)prv;
	if (d->f) fclose(d->f);
	decoder_error_clear(&d->error);
	free(d);
}

static int dsd_decode(void *prv, char *buf, int buf_len,
                      struct sound_params *sp)
{
	struct dsd_data *d = (struct dsd_data *)prv;
	if (!d->ok) return 0;

	sp->channels = d->channels;
	sp->rate     = d->pcm_rate;

	switch (d->mode) {
	case DSD_MODE_NATIVE:
		/* Raw DSD bytes in the container width/endianness the DAC needs.
		 * native_fmt is one of SFMT_DSD_U8, SFMT_DSD_U16|SFMT_LE/BE,
		 * SFMT_DSD_U32|SFMT_LE/BE — no PCM endianness bits appended. */
		sp->fmt = d->native_fmt;
		return decode_native(d, buf, buf_len);

	case DSD_MODE_DOP:
		/* 24-bit DoP payload in S32_LE words; standard PCM path. */
		sp->fmt = SFMT_S32 | SFMT_LE;
		return decode_dop(d, buf, buf_len);

	case DSD_MODE_PCM:
		sp->fmt = SFMT_FLOAT;
		return decode_pcm(d, buf, buf_len);
	}
	return 0;
}

static int dsd_seek(void *prv, int sec)
{
	struct dsd_data *d = (struct dsd_data *)prv;
	if (!d->ok || sec < 0) return -1;

	uint64_t bps = (uint64_t)(d->dsd_rate / 8) * d->channels;
	uint64_t target = (uint64_t)sec * bps;

	/* Align to container-word boundary for native mode */
	int align = (d->mode == DSD_MODE_NATIVE) ? d->native_Bpc : 1;

	if (d->format == DSD_FMT_DSF) {
		uint64_t blk = d->dsf_block_size;
		uint64_t bytes_per_ch = target / d->channels;
		bytes_per_ch = (bytes_per_ch / blk) * blk;
		/* also align to container width */
		bytes_per_ch = (bytes_per_ch / align) * align;
		target = bytes_per_ch * d->channels;
		fseek(d->f, d->data_offset + (long)bytes_per_ch, SEEK_SET);
	} else {
		target = (target / (align * d->channels)) * (align * d->channels);
		fseek(d->f, d->data_offset + (long)target, SEEK_SET);
	}

	d->bytes_decoded = target;
	d->dop_phase     = 0;
	d->fir_pos       = 0;
	memset(d->fir_buf, 0, sizeof(d->fir_buf));

	return bps > 0 ? (int)(target / bps) : -1;
}

static void dsd_info(const char *file, struct file_tags *tags,
                     const int tags_sel)
{
	if (tags_sel & TAGS_TIME) {
		struct dsd_data *tmp = (struct dsd_data *)dsd_open(file);
		if (tmp && tmp->ok) tags->time = tmp->duration;
		if (tmp) dsd_close(tmp);
	}
}

static int dsd_get_bitrate(void *prv)
{
	struct dsd_data *d = (struct dsd_data *)prv;
	return (int)(d->dsd_rate / 1000);
}

static int dsd_get_duration(void *prv)
{
	struct dsd_data *d = (struct dsd_data *)prv;
	return d->ok ? d->duration : -1;
}

static void dsd_get_error(void *prv, struct decoder_error *error)
{
	decoder_error_copy(error, &((struct dsd_data *)prv)->error);
}

static int dsd_our_format_ext(const char *ext)
{
	return !strcasecmp(ext, "dsf") || !strcasecmp(ext, "dff");
}

static void dsd_get_name(const char *file, char buf[4])
{
	const char *ext = ext_pos(file);
	if (ext) {
		if      (!strcasecmp(ext, "dsf")) strcpy(buf, "DSF");
		else if (!strcasecmp(ext, "dff")) strcpy(buf, "DFF");
		else                              strcpy(buf, "DSD");
	}
}

/* =========================================================================
 * Plugin descriptor
 * ====================================================================== */
static struct decoder dsd_decoder = {
	DECODER_API_VERSION,
	NULL, NULL,
	dsd_open, NULL, NULL,
	dsd_close, dsd_decode, dsd_seek, dsd_info,
	dsd_get_bitrate, dsd_get_duration, dsd_get_error,
	dsd_our_format_ext, NULL, dsd_get_name,
	NULL, NULL,
	dsd_get_bitrate /* avg_bitrate same as bitrate */
};

struct decoder *plugin_init(void)
{
	return &dsd_decoder;
}
