/*
 * This file is part of the demodulator distribution
 * (https://github.com/peads/demodulator).
 * with code originally part of the misc_snippets distribution
 * (https://github.com/peads/misc_snippets).
 * Copyright (c) 2023 Patrick Eads.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <http://www.gnu.org/licenses/>.
 */
#include <unistd.h>
#include <stdlib.h>
#include "matrix.h"

#ifdef __APPLE__
#define SEM_INIT(SEM, NAME, VALUE) \
    args.exitFlag |= printIfError( \
        (SEM = sem_open (NAME, O_CREAT | O_EXCL, 0644, VALUE)));
#else
#define SEM_INIT(SEM, NAME, VALUE) \
    SEM = malloc(sizeof(sem_t)); \
    args.exitFlag |= printIfError( \
        sem_init(SEM, 0, VALUE) ? NULL : SEM);
#endif
#ifdef __APPLE__
#define SEM_DESTROY(SEM, NAME) \
    sem_close(SEM); \
    sem_unlink(NAME);
#else
#define SEM_DESTROY(SEM, NAME) \
    sem_destroy(SEM); \
    free(SEM);
#endif

#ifdef IS_NVIDIA
extern void *processMatrix(void *ctx);
extern void allocateBuffer(void **buf,  size_t len);
#endif

static inline int printIfError(void *file) {

    if (!file) {
        perror(NULL);
        return 1;
    }
    return 0;
}

static inline int startProcessingMatrix(
        FILE *inFile,
        consumerArgs *args) {

    size_t elementsRead;
    pthread_t pid;

    allocateBuffer(&args->buf, args->bufSize);

    args->exitFlag |= printIfError(
            pthread_create(&pid, NULL, processMatrix, args)
            ? NULL
            : &args);

    while (!args->exitFlag) {

        sem_wait(args->empty);
        pthread_mutex_lock(&args->mutex);
        elementsRead = fread(args->buf, 1, args->bufSize, inFile);

        if ((args->exitFlag = ferror(inFile))) {
            perror(NULL);
            args->exitFlag = -2;
            break;
        } else if (feof(inFile)) {
            args->exitFlag = EOF;
        } else if (!elementsRead) {
            args->exitFlag = -3;
            break;
        }
        args->bufSize = elementsRead;
        pthread_mutex_unlock(&args->mutex);
        sem_post(args->full);
    }

    pthread_join(pid, NULL);
    pthread_mutex_destroy(&args->mutex);
    fclose(args->outFile);
    fclose(inFile);
    return args->exitFlag != EOF;
}

int main(int argc, char **argv) {

    consumerArgs args = {
            .bufSize = DEFAULT_BUF_SIZE,
            .mutex = PTHREAD_MUTEX_INITIALIZER,
            .sampleRate = 10.,
            .lowpassIn = 0.,
            .lowpassOut = 1.,
            .inFilterDegree = 0,
            .outFilterDegree = 5,
            .epsilon = .3,
            .exitFlag = 0,
            .mode = 0x10
    };
    SEM_INIT(args.empty, "/empty", 1)
    SEM_INIT(args.full, "/full", 0)

    int ret = 0;
    int opt;
    long bufShift;
    FILE *inFile = NULL;

    if (argc < 3) {
        return -1;
    } else {
        while ((opt = getopt(argc, argv, "i:o:r:L:l:S:D:d:e:m:b:c:q:w:")) != -1) {
            switch (opt) {
                case 'i':
                    if (!strstr(optarg, "-")) {
                        ret += printIfError(inFile = fopen(optarg, "rb"));
                    } else {
                        ret += printIfError(freopen(NULL, "rb", stdin));
                        inFile = stdin;
                    }
                    break;
                case 'o':
                    if (!strstr(optarg, "-")) {
                        ret += printIfError(args.outFile = fopen(optarg, "wb"));
                    } else {
                        ret += printIfError(freopen(NULL, "wb", stdout));
                        args.outFile = stdout;
                    }
                    break;
                case 'L':
                    args.lowpassIn = TO_REAL(optarg, NULL);
                    break;
                case 'l':
                    args.lowpassOut = TO_REAL(optarg, NULL);
                    break;
                case 'S':
                    args.sampleRate = TO_REAL(optarg, NULL);
                    break;
                case 'D':
                    args.inFilterDegree = strtoul(optarg, NULL, 10);
                    break;
                case 'd':
                    args.outFilterDegree = strtoul(optarg, NULL, 10);
                    break;
                case 'e':
                    args.epsilon = TO_REAL(optarg, NULL) / 10.;
                    break;
                case 'm':
                    args.mode |= strtoul(optarg, NULL, 10);
                    break;
                case 'b':
                    bufShift = strtol(optarg, NULL, 10);
                    if (args.bufSize && labs(bufShift) < 17) {
                        if (bufShift < 1) {
                            args.bufSize = DEFAULT_BUF_SIZE >> -bufShift;
                        } else {
                            args.bufSize = DEFAULT_BUF_SIZE << bufShift;
                        }
                    }
                    break;
                case 'c':
                    args.mode |= strtoul(optarg, NULL, 10) << 4;
                    break;
                case 'q':
                    args.mode |= strtoul(optarg, NULL, 10) << 2;
                    break;
                case 'w':
                    args.mode |= strtoul(optarg, NULL, 10) << 6;
                    break;
                default:
                    break;
            }
        }
    }

    if (!ret) {
        startProcessingMatrix(inFile, &args);
    }

    SEM_DESTROY(args.empty, "/empty")
    SEM_DESTROY(args.full, "/full")
    return ret;
}
