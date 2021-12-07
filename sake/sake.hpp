#pragma once

void sake_malloc(uint8_t** run, Message** msgs);
void sake_free(uint8_t* run, Message* msgs);
void sake_runner(uint8_t* run, Message* msgs);