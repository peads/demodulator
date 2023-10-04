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

int printIfError(FILE *file) {

    if (!file) {
        perror(NULL);
        return 1;
    }
    return 0;
}

int main(int argc, char **argv) {

    float gain = 1.f;
    uint8_t mode = 0;
    int ret = 0;
    int opt;
    FILE *inFile = NULL;
    FILE *outFile = NULL;
    pthread_t pid;
    size_t elementsRead;

    if (argc < 3) {
        return -1;
    } else {
        while ((opt = getopt(argc, argv, "g:i:o:r:")) != -1) {
            switch (opt) {
                case 'g':
                    gain = strtof(optarg, NULL);
                    break;
                case 'r' :
                    mode = strtol(optarg, NULL, 10);
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
                default:
                    break;
            }
        }
    }

    if (ret) {
        return ret;
    }
    consumerArgs args = {
            .mutex = PTHREAD_MUTEX_INITIALIZER,
            .mode = mode,
            .outFile = outFile,
            .exitFlag = sem_init(&args.empty, 0, 1),
            .gain = gain != 1.f ? gain : 0.f
    };
    allocateBuffer(&args.buf, DEFAULT_BUF_SIZE);

    args.exitFlag |= sem_init(&args.full, 0, 0);
    args.exitFlag |= pthread_create(&pid, NULL, processMatrix, &args);

    while (!args.exitFlag) {

        sem_wait(&args.empty);
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
        sem_post(&args.full);
    }


    pthread_join(pid, NULL);
    pthread_mutex_destroy(&args.mutex);
    sem_destroy(&args.empty);
    sem_destroy(&args.full);

    fclose(outFile);
    fclose(inFile);
    return args.exitFlag != EOF;
}
