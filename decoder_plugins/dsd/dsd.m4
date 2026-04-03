dnl DSD decoder plugin (DSF .dsf and DSDIFF .dff files)
dnl No external library required — pure C implementation.

AC_ARG_WITH(dsd, AS_HELP_STRING([--without-dsd],
                                [Compile without DSD (DSF/DFF) support]))

if test "x$with_dsd" != "xno"
then
	want_dsd="yes"
	DECODER_PLUGINS="$DECODER_PLUGINS dsd"
fi

AM_CONDITIONAL([BUILD_dsd], [test "$want_dsd"])
AC_CONFIG_FILES([decoder_plugins/dsd/Makefile])
