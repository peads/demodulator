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
#include <string.h>
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
        const float lowpassIn, float highpassIn,
        const float gain, FILE *outFile) {

    size_t elementsRead;
    pthread_t pid;

    consumerArgs args = {
            .mutex = PTHREAD_MUTEX_INITIALIZER,
            .lowpassIn = lowpassIn,
            .highpassIn = highpassIn,
            .outFile = outFile,
            .exitFlag = 0,
            .gain = gain != 1.f ? gain : 0.f
    };

    SEM_INIT(args.empty, "/empty", 1)
    SEM_INIT(args.full, "/full", 0)

    args.exitFlag |= printIfError(
            pthread_create(&pid, NULL, processMatrix, &args)
            ? NULL
            : &args);

    allocateBuffer(&args.buf, DEFAULT_BUF_SIZE);

    while (!args.exitFlag) {

        sem_wait(args.empty);
        pthread_mutex_lock(&args.mutex);
        elementsRead = fread(args.buf, 1, DEFAULT_BUF_SIZE, inFile);

        if ((args.exitFlag = ferror(inFile))) {
            perror(NULL);
            break;
        } else if (feof(inFile)) {
            args.exitFlag = EOF;
        } else if (!elementsRead) {
            fprintf(stderr, "This shouldn't happen, but I need to use the result of"
                            "fread. Stupid compiler.");
        }
        pthread_mutex_unlock(&args.mutex);
        sem_post(args.full);
    }

    pthread_join(pid, NULL);
    pthread_mutex_destroy(&args.mutex);
    SEM_DESTROY(args.empty, "/empty")
    SEM_DESTROY(args.full, "/full")

    fclose(outFile);
    fclose(inFile);
    return args.exitFlag != EOF;
}

int main(int argc, char **argv) {

    float gain = 1.f;
    float lowpassIn = 0.f;
    float highpassIn = 0.f;
    int ret = 0;
    int opt;
    FILE *inFile = NULL;
    FILE *outFile = NULL;

    if (argc < 3) {
        return -1;
    } else {
        while ((opt = getopt(argc, argv, "g:i:o:r:l:L:h:")) != -1) {
            switch (opt) {
                case 'g':
                    gain = strtof(optarg, NULL);
                    break;
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
                        ret += printIfError(outFile = fopen(optarg, "wb"));
                    } else {
                        ret += printIfError(freopen(NULL, "wb", stdout));
                        outFile = stdout;
                    }
                    break;
                case 'l':
                    if (!lowpassIn) {
                        lowpassIn = strtof(optarg, NULL);
                        break;
                    }
                    return -1;
                case 'L':
                    if (!lowpassIn) {
                        lowpassIn = 1.f/strtof(optarg, NULL);
                        break;
                    }
                    return -1;
                case 'h':
                    if (!highpassIn) {
                        highpassIn = strtof(optarg, NULL);
                        break;
                    }
                    return -1;
                default:
                    break;
            }
        }
    }

    if (!ret) {
        startProcessingMatrix(inFile, lowpassIn, highpassIn, gain, outFile);
    }
    return ret;
}
