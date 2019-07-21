#include <stdbool.h>

typedef void* ALEInterface;
typedef void* ALEState;

// Declares int rgb_palette[256]
ALEInterface *ALE_new();
void ALE_del(ALEInterface *ale);

const char *getString(ALEInterface *ale, const char *key);
int getInt(ALEInterface *ale,const char *key);
bool getBool(ALEInterface *ale,const char *key);
float getFloat(ALEInterface *ale,const char *key);

void setString(ALEInterface *ale,const char *key,const char *value);
void setInt(ALEInterface *ale,const char *key,int value);
void setBool(ALEInterface *ale,const char *key,bool value);
void setFloat(ALEInterface *ale,const char *key,float value);

void loadROM(ALEInterface *ale,const char *rom_file);
int act(ALEInterface *ale,int action);
void reset_game(ALEInterface *ale);
bool game_over(ALEInterface *ale);
int lives(ALEInterface *ale);
int getFrameNumber(ALEInterface *ale);
int getEpisodeFrameNumber(ALEInterface *ale);

void getAvailableModes(ALEInterface *ale,int *availableModes);
int getAvailableModesSize(ALEInterface *ale);
void setMode(ALEInterface *ale, int mode);
void getAvailableDifficulties(ALEInterface *ale,int *availableDifficulties);
int getAvailableDifficultiesSize(ALEInterface *ale);
void setDifficulty(ALEInterface *ale, int difficulty);
void getLegalActionSet(ALEInterface *ale,int *actions);
int getLegalActionSize(ALEInterface *ale);
void getMinimalActionSet(ALEInterface *ale,int *actions);
int getMinimalActionSize(ALEInterface *ale);

int getScreenWidth(ALEInterface *ale);
int getScreenHeight(ALEInterface *ale);
void getScreen(ALEInterface *ale,unsigned char *screen_data);
void getScreenRGB(ALEInterface *ale, unsigned char *output_buffer);
void getScreenGrayscale(ALEInterface *ale, unsigned char *output_buffer);
void saveScreenPNG(ALEInterface *ale,const char *filename);

void getRAM(ALEInterface *ale,unsigned char *ram);
int getRAMSize(ALEInterface *ale);

void saveState(ALEInterface *ale);
void loadState(ALEInterface *ale);
ALEState* cloneState(ALEInterface *ale);
void restoreState(ALEInterface *ale, ALEState* state);
ALEState* cloneSystemState(ALEInterface *ale);
void restoreSystemState(ALEInterface *ale, ALEState* state);
void deleteState(ALEState* state);

// Encodes the state as a raw bytestream. This may have multiple '\0' characters
// and thus should not be treated as a C string. Use encodeStateLen to find the length
// of the buffer to pass in, or it will be overrun as this simply memcpys bytes into the buffer.
void encodeState(ALEState *state, char *buf, int buf_len);
int encodeStateLen(ALEState *state);
ALEState *decodeState(const char *serialized, int len);

// 0: Info, 1: Warning, 2: Error
void setLoggerMode(int mode);
