//
// Created by peads on 4/4/23.
//
#include "matrix.h"
int8_t processMatrix(float squelch, FILE *inFile, struct chars *chars, FILE *outFile) {

    uint8_t buf[DEFAULT_BUF_SIZE];
    float result[DEFAULT_BUF_SIZE<<2]; // TODO revrt back to QTR_BUF_SIZE

    int8_t exitFlag = 0;
    size_t readBytes;

    while (!exitFlag) {

        readBytes = fread(buf, INPUT_ELEMENT_BYTES, DEFAULT_BUF_SIZE, inFile);

        if (exitFlag = ferror(inFile)) {
            perror(NULL);
            break;
        } else if (feof(inFile)) {
            exitFlag = EOF;
        }

        fmDemod(buf, readBytes, result);

        fwrite(result, OUTPUT_ELEMENT_BYTES, QTR_BUF_SIZE, outFile);
    }
    return exitFlag;
}