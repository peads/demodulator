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
            .mutex = PTHREAD_MUTEX_INITIALIZER,
            .sampleRate = 0.f,
            .lowpassIn = 0.f,
            .lowpassOut = 0.f,
            .inFilterDegree = 0,
            .outFilterDegree = 0,
            .epsilon = 0.f,
            .exitFlag = 0,
            .mode = 2,
            .bufSize = DEFAULT_BUF_SIZE
    };
    SEM_INIT(args.empty, "/empty", 1)
    SEM_INIT(args.full, "/full", 0)

    int ret = 0;
    int opt;
    FILE *inFile = NULL;

    if (argc < 3) {
        return -1;
    } else {
        while ((opt = getopt(argc, argv, "i:o:r:L:l:S:D:d:e:m:b:")) != -1) {
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
                    args.lowpassIn = strtof(optarg, NULL);
                    break;
                case 'l':
                    args.lowpassOut = strtof(optarg, NULL);
                    break;
                case 'S':
                    args.sampleRate = strtof(optarg, NULL);
                    break;
                case 'D':
                    // TODO re-separate these for different input and output degrees
//                    args.inFilterDegree = strtol(optarg, NULL, 10);
//                    break;
                case 'd':
                    args.outFilterDegree = strtol(optarg, NULL, 10);
                    break;
                case 'e':
                    args.epsilon = strtof(optarg, NULL) / 10.f;
                    break;
                case 'm':
                    args.mode = strtol(optarg, NULL, 10);
                    break;
                case 'b':
                    args.bufSize = strtol(optarg, NULL, 10);
                    args.bufSize = args.bufSize < 1 || args.bufSize > 5 ? DEFAULT_BUF_SIZE : DEFAULT_BUF_SIZE << args
                            .bufSize;
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
