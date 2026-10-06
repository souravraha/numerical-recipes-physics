/* Tiny single-translation-unit test harness (one per test_*.c). */
#ifndef NR_TEST_HARNESS_H
#define NR_TEST_HARNESS_H

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static int nr_checks_run = 0;
static int nr_checks_failed = 0;

#define CHECK(cond)                                                       \
    do {                                                                  \
        nr_checks_run++;                                                  \
        if (!(cond)) {                                                    \
            nr_checks_failed++;                                           \
            fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
        }                                                                 \
    } while (0)

#define CHECK_NEAR(a, b, tol)                                             \
    do {                                                                  \
        nr_checks_run++;                                                  \
        double _a = (a), _b = (b);                                        \
        if (!(fabs(_a - _b) <= (tol))) {                                  \
            nr_checks_failed++;                                           \
            fprintf(stderr, "FAIL %s:%d: |%.12g - %.12g| <= %.3g\n",     \
                    __FILE__, __LINE__, _a, _b, (double)(tol));           \
        }                                                                 \
    } while (0)

/* Report and return exit code. */
static int nr_report(const char *suite) {
    if (nr_checks_failed) {
        fprintf(stderr, "%s: %d/%d checks FAILED\n", suite, nr_checks_failed, nr_checks_run);
        return 1;
    }
    printf("%s: %d checks passed\n", suite, nr_checks_run);
    return 0;
}

#endif
