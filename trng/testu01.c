// Compile:
// gcc testu01.c -I $HOME/code/TestU01-install/include $HOME/code/TestU01-install/lib/libtestu01.a $HOME/code/TestU01-install/lib/libprobdist.a  $HOME/code/TestU01-install/lib/libmylib.a -lm -o testu01
// Run:
// ./testu01 random.bin > output.txt
// Postprocess output:
// sed -n '/Summary/,/End/p' output.txt
#include "TestU01.h"
#include <stdlib.h>

char* buffer;
long bytes;
long pos;

unsigned char gen8() {
    char value = buffer[pos];
    pos = (pos + 1) % bytes;
    return value;
}

unsigned int readInt() {
    unsigned res = 0;
    for (int i = 0; i < sizeof(int); i++) {
        unsigned byte = gen8();
        res = res << 8;
        res = res | byte;
    }
    return res;
}

int main(int argc, char** argv) {
    char* filename = "random.bin";
    if (argc >= 2) {
        filename = argv[1];
    }

    FILE* fp = fopen(filename, "rb");
    fseek(fp, 0, SEEK_END);        
    bytes = ftell(fp);           
    rewind(fp);                    

    buffer = (char*) malloc(bytes);
    fread(buffer, sizeof(char), bytes, fp);
    fclose(fp);

    pos = 0;

    unif01_Gen* gen = unif01_CreateExternGenBits("file_gen", readInt);

    //swrite_Basic = FALSE;
    //swrite_Parameters = FALSE;
    //swrite_Collectors = FALSE;
    //swrite_Classes = FALSE;
    //swrite_Counters = FALSE;
    //swrite_Host = FALSE;


    bbattery_pseudoDIEHARD(gen);
    printf("End of test\n");
    bbattery_FIPS_140_2(gen);
    printf("End of test\n");
    bbattery_SmallCrush(gen);
    printf("End of test\n");
    bbattery_Alphabit(gen, 1 << 26, 0, 32);
    printf("End of test\n");
    bbattery_Rabbit(gen, 1 << 26);
    printf("End of test\n");
    //bbattery_Crush(gen);
    //printf("End of test\n");
    //bbattery_BigCrush(gen);
    //printf("End of test\n");
}
