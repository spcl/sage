#pragma once

void sake_malloc(uint8_t** run, Message** msgs, Message** rsps);
void sake_free(uint8_t* run, Message* msgs, Message* rsps);
void sake_runner(uint8_t* run, Message* msgs, Message* rsps);