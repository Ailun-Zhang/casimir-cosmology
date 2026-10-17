/* PROVENANCE: 04_calculations/code/20261014_claude_v12_r3/sm_mapping/mingw_stubs/execinfo.h (copied 2026-10-15 by code-higgs) */
/* C148 run02 (sm-mapping): MinGW has no glibc <execinfo.h>.  TSIL/3VIL/SMDR use backtrace() only to print a
   stack trace inside their fatal-error handlers; these no-op stubs return an empty trace. */
#ifndef C148_EXECINFO_STUB_H
#define C148_EXECINFO_STUB_H
#include <stddef.h>
static inline int backtrace(void **buffer, int size) { (void) buffer; (void) size; return 0; }
static inline char **backtrace_symbols(void *const *buffer, int size) { (void) buffer; (void) size; return NULL; }
static inline void backtrace_symbols_fd(void *const *buffer, int size, int fd) { (void) buffer; (void) size; (void) fd; }
#endif
