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
//int8_t processMatrix(float squelch, FILE *inFile, struct chars *chars, FILE *outFile) {
//
//    uint8_t buf[DEFAULT_BUF_SIZE];
//    float result[QTR_BUF_SIZE];
//
//    int8_t exitFlag = 0;
//    size_t readBytes;
//
//    while (!exitFlag) {
//
//        readBytes = fread(buf, INPUT_ELEMENT_BYTES, DEFAULT_BUF_SIZE, inFile);
//
//        if (exitFlag = ferror(inFile)) {
//            perror(NULL);
//            break;
//        } else if (feof(inFile)) {
//            exitFlag = EOF;
//        }
//
//        fmDemod(buf, readBytes, result);
//
//        fwrite(result, OUTPUT_ELEMENT_BYTES, QTR_BUF_SIZE, outFile);
//    }
//    return exitFlag;
//}