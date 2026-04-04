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
#define FIR_LEN            127  /* 127-tap sinc-Blackman, fc=0.05 (~17640 Hz @ 352800 Hz rate) */
#define DSD_READ_BYTES     65536  /* large enough for ratio=256: 65536/256=256 samples/call */

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
	-0.00000000f, 0.00000068f, 0.00000145f, -0.00000000f, -0.00000605f, -0.00001845f,
	-0.00003755f, -0.00006182f, -0.00008750f, -0.00010872f, -0.00011802f, -0.00010740f,
	-0.00006963f, 0.00000000f, 0.00010211f, 0.00023172f, 0.00037741f, 0.00052124f,
	0.00063977f, 0.00070619f, 0.00069359f, 0.00057906f, 0.00034831f, -0.00000000f,
	-0.00045063f, -0.00096980f, -0.00150542f, -0.00198996f, -0.00234620f, -0.00249555f,
	-0.00236840f, -0.00191545f, -0.00111859f, 0.00000000f, 0.00137237f, 0.00288406f,
	0.00437917f, 0.00567172f, 0.00656263f, 0.00686163f, 0.00641174f, 0.00511418f,
	0.00295064f, -0.00000000f, -0.00355315f, -0.00742032f, -0.01122220f, -0.01451324f,
	-0.01681601f, -0.01766311f, -0.01664282f, -0.01344410f, -0.00789647f, 0.00000000f,
	0.01005800f, 0.02190252f, 0.03498742f, 0.04862874f, 0.06205209f, 0.07445049f,
	0.08504754f, 0.09316004f, 0.09825441f, 0.09999134f, 0.09825441f, 0.09316004f,
	0.08504754f, 0.07445049f, 0.06205209f, 0.04862874f, 0.03498742f, 0.02190252f,
	0.01005800f, 0.00000000f, -0.00789647f, -0.01344410f, -0.01664282f, -0.01766311f,
	-0.01681601f, -0.01451324f, -0.01122220f, -0.00742032f, -0.00355315f, -0.00000000f,
	0.00295064f, 0.00511418f, 0.00641174f, 0.00686163f, 0.00656263f, 0.00567172f,
	0.00437917f, 0.00288406f, 0.00137237f, 0.00000000f, -0.00111859f, -0.00191545f,
	-0.00236840f, -0.00249555f, -0.00234620f, -0.00198996f, -0.00150542f, -0.00096980f,
	-0.00045063f, -0.00000000f, 0.00034831f, 0.00057906f, 0.00069359f, 0.00070619f,
	0.00063977f, 0.00052124f, 0.00037741f, 0.00023172f, 0.00010211f, 0.00000000f,
	-0.00006963f, -0.00010740f, -0.00011802f, -0.00010872f, -0.00008750f, -0.00006182f,
	-0.00003755f, -0.00001845f, -0.00000605f, -0.00000000f, 0.00000145f, 0.00000068f,
	-0.00000000f
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
	int          native_Bpc;
	int          native_swap;

	/* FIR state (PCM mode) */
	float        fir_buf[DSD_MAX_CHANNELS][FIR_LEN];
	int          fir_pos;
	int          pcm_decimate; /* decimation ratio: dsd_rate/8 for all formats */

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
			int filled = 0;
			uint64_t bpc = bytes_per_ch;
			while (filled < req) {
				int offset_in_block = (int)(bpc % blk);
				int avail = blk - offset_in_block;
				int want  = req - filled;
				if (want > avail) want = avail;
				long pos = d->data_offset
				           + (long)((bpc / blk) * ((uint64_t)blk * d->channels))
				           + (long)(ch * blk)
				           + offset_in_block;
				fseek(d->f, pos, SEEK_SET);
				int r = (int)fread(buf[ch] + filled, 1, want, d->f);
				if (r <= 0) {
					if (filled == 0) return 0;
					memset(buf[ch] + filled, 0, req - filled);
					filled = req;
					break;
				}
				filled += r;
				bpc    += r;
			}
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
			uint8_t b[4];
			for (int k = 0; k < Bpc; k++)
				b[k] = bit_rev[raw[ch][i * Bpc + k]];

			if (want_be) {
				/* DSD_U32_BE / DSD_U16_BE: bit-reversed bytes in
				 * chronological order (oldest first = MSB).
				 * Matches squeezelite DSD_U32_BE output for LSB-first DSF. */
				for (int k = 0; k < Bpc; k++)
					*p++ = b[k];
			} else {
				/* DSD_U32_LE / DSD_U16_LE: bit-reversed bytes in
				 * reverse chronological order (newest byte at LSB). */
				for (int k = Bpc - 1; k >= 0; k--)
					*p++ = b[k];
			}
		}
	}

	d->bytes_decoded += (uint64_t)frames_got * Bpc * d->channels;
	return frames_got * Bpc * d->channels;
}

/* =========================================================================
 * MODE: DoP  (DSD-over-PCM, IEC 62236-3)
 *
 * Packing (matches ffmpeg/mpv reference implementation):
 *   Logical S32BE word: [marker][dsd_byte0][dsd_byte1][0x00]
 *   In S32LE memory:    [0x00][dsd_byte1][dsd_byte0][marker]
 *
 * DSF files are LSB-first: each byte is bit-reversed before packing.
 * DSDIFF files are MSB-first: no bit-reversal needed.
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
			uint8_t b0 = raw[ch][i];       /* first DSD byte from file  */
			uint8_t b1 = raw[ch][i + 1];   /* second DSD byte from file */

			/* DSF is LSB-first: bit-reverse to get MSB-first DSD bytes.
			 * DSDIFF is already MSB-first. */
			if (d->lsb_first) { b0 = bit_rev[b0]; b1 = bit_rev[b1]; }

			/* S32LE memory layout: [0x00][b1][b0][marker]
			 * = logical S32BE:     [marker][b0][b1][0x00]
			 * This matches the ffmpeg/mpv DoP reference implementation. */
			*p++ = 0x00u;
			*p++ = b1;
			*p++ = b0;
			*p++ = marker;
		}
		frames_out++;
	}

	d->bytes_decoded += (uint64_t)got * d->channels;
	return frames_out * 4 * d->channels;
}

/* =========================================================================
 * MODE: PCM  (software FIR decimation)  → SFMT_FLOAT
 *
 * The FIR filter operates at the DSD byte level:
 *   - Each DSD byte is converted to a "pulse density" value in [-1, +1]
 *     by counting set bits: density = (popcount(byte) * 2 - 8) / 8.0
 *   - The delay line stores one float per DSD byte (not per bit).
 *   - One output PCM sample is produced per input DSD byte per channel.
 *   - Output rate = dsd_rate / 8 (e.g. DSD64 → 352800 Hz).
 *
 * The decimation ratio (d->pcm_decimate) is chosen so the output rate
 * stays at or below 352800 Hz regardless of DSD rate:
 *   DSD64  (2822400): ratio=8  → 352800 Hz
 *   DSD128 (5644800): ratio=16 → 352800 Hz
 *   DSD256 (11289600):ratio=32 → 352800 Hz
 * ====================================================================== */
/* =========================================================================
 * MODE: PCM  (software DSD→float via FIR low-pass)
 *
 * One output float per DSD byte per channel (rate = dsd_rate/8).
 * Each DSD byte is converted to a pulse-density value [-1,+1] via a
 * popcount LUT, fed into a 63-tap FIR low-pass filter (fc = 0.1 * byte_rate),
 * and the filtered value is the output PCM sample.
 *
 * This is the same approach as MPD's software DSD decoder.
 * ====================================================================== */
static int decode_pcm(struct dsd_data *d, char *out, int out_len)
{
	int samples_wanted = out_len / ((int)sizeof(float) * d->channels);
	if (samples_wanted <= 0) return 0;
	if (samples_wanted > DSD_READ_BYTES) samples_wanted = DSD_READ_BYTES;

	uint64_t bytes_per_ch_left =
		(d->total_dsd_bytes - d->bytes_decoded) / d->channels;
	if (bytes_per_ch_left == 0) return 0;
	if ((uint64_t)samples_wanted > bytes_per_ch_left)
		samples_wanted = (int)bytes_per_ch_left;
	if (samples_wanted <= 0) return 0;

	static uint8_t raw[DSD_MAX_CHANNELS][DSD_READ_BYTES];
	int got = dsd_read(d, raw, samples_wanted);
	if (got <= 0) return 0;

	/* density LUT: byte value → pulse density in [-1.0, +1.0]
	 * density = (popcount(byte) * 2 - 8) / 8.0 */
	static float density_lut[256];
	static int   lut_ready = 0;
	if (!lut_ready) {
		for (int b = 0; b < 256; b++)
			density_lut[b] = (__builtin_popcount(b) * 2 - 8) / 8.0f;
		lut_ready = 1;
	}

	float *fout = (float *)out;
	int samples_out = 0;

	for (int i = 0; i < got; i++) {
		/* Push one density value per channel into the FIR delay line */
		for (int ch = 0; ch < d->channels; ch++) {
			uint8_t byte = raw[ch][i];
			if (d->lsb_first) byte = bit_rev[byte];
			d->fir_buf[ch][d->fir_pos] = density_lut[byte];
		}
		d->fir_pos = (d->fir_pos + 1) % FIR_LEN;

		/* Convolve FIR for each channel → one PCM sample */
		for (int ch = 0; ch < d->channels; ch++) {
			float acc = 0.0f;
			for (int k = 0; k < FIR_LEN; k++) {
				int idx = ((int)d->fir_pos - 1 - k + FIR_LEN * 4) % FIR_LEN;
				acc += dsd_fir[k] * d->fir_buf[ch][idx];
			}
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
			d->native_fmt = SFMT_DSD_U16 | SFMT_BE;
			d->native_Bpc = 2;
		} else if (nf && !strcasecmp(nf, "u32le")) {
			d->native_fmt = SFMT_DSD_U32 | SFMT_LE;
			d->native_Bpc = 4;
		} else if (nf && !strcasecmp(nf, "u32be")) {
			/* DAC natively supports DSD_U32_BE — use it directly,
			 * no byte-swap needed. */
			d->native_fmt = SFMT_DSD_U32 | SFMT_BE;
			d->native_Bpc = 4;
		} else {
			/* default: u8 */
			d->native_fmt = SFMT_DSD_U8;
			d->native_Bpc = 1;
		}

		logit("DSD: native format=%s Bpc=%d",
		      d->native_fmt == SFMT_DSD_U8              ? "DSD_U8"     :
		      d->native_fmt == (SFMT_DSD_U16 | SFMT_LE) ? "DSD_U16_LE" :
		      d->native_fmt == (SFMT_DSD_U16 | SFMT_BE) ? "DSD_U16_BE" :
		      d->native_fmt == (SFMT_DSD_U32 | SFMT_LE) ? "DSD_U32_LE" :
		      d->native_fmt == (SFMT_DSD_U32 | SFMT_BE) ? "DSD_U32_BE" : "?",
		      d->native_Bpc);
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
		/* Output one float per DSD byte per channel.
		 * Rate = dsd_rate / 8, same as MPD's software DSD decoder.
		 * If the DAC doesn't support this rate, audio_conversion will
		 * resample to the nearest supported rate. */
		d->pcm_decimate = 1;
		d->pcm_rate     = d->dsd_rate / 8;
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
	sp->dsd_rate = d->dsd_rate;

	switch (d->mode) {
	case DSD_MODE_NATIVE:
		/* Raw DSD bytes in the container width/endianness the DAC needs.
		 * native_fmt is one of SFMT_DSD_U8, SFMT_DSD_U16|SFMT_LE/BE,
		 * SFMT_DSD_U32|SFMT_LE/BE — no PCM endianness bits appended. */
		sp->fmt = d->native_fmt;
		return decode_native(d, buf, buf_len);

	case DSD_MODE_DOP:
		/* DoP: S32_LE container with DSD+marker. SFMT_DOP protects
		 * from softmixer/equalizer/conversion. ALSA strips the flag. */
		sp->fmt = SFMT_DOP | SFMT_S32 | SFMT_LE;
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
	/* Report DSD bit-rate per channel in kbps.
	 * DSD64:  2822 kbps/ch, DSD128: 5644 kbps/ch, DSD256: 11289 kbps/ch.
	 * The display caps at 9999, so DSD256 shows as 9999 — acceptable. */
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
	if (!(!strcasecmp(ext, "dsf") || !strcasecmp(ext, "dff")))
		return 0;

	/* In PCM mode, let ffmpeg handle DSD→PCM conversion — it uses
	 * the proven dsd2pcm library and correctly handles all DSD rates. */
	const char *mode = options_get_str("DSDPlaybackMode");
	if (mode && !strcasecmp(mode, "pcm"))
		return 0;

	return 1;
}

static void dsd_get_name(const char *file, char buf[4])
{
	/* buf is exactly 4 bytes: max 3 chars + null terminator.
	 * Show the container format: DSF or DFF. */
	const char *ext = ext_pos(file);
	if (ext) {
		if      (!strcasecmp(ext, "dsf")) strcpy(buf, "DSF");
		else if (!strcasecmp(ext, "dff")) strcpy(buf, "DFF");
		else                              strcpy(buf, "DSD");
	} else {
		strcpy(buf, "DSD");
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
