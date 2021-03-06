/*Tyler j Roberts
  CS441
  brute_cpu.cu
  Template Made by Dr. Mock
/*

/***********************************************************************
 * brute_cpu.cu
 *
 * Brute force crack of an MD5 hash.  This program assumes the string key
 * is exactly 6 letters long and all uppercase letters (it is not too much
 * additional work to handle arbritrary symbols and a variable number of
 * letters).  
 *
 * It cracks a MD5 hash, currently hard-coded in main, by trying all
 * six letter strings from AAAAAA to ZZZZZZ, hashing each one, and checking
 * if it matches the target hash.
 *
 ***********************************************************************/
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "md5.cu"
 
 // Convert a decimal number (starting at 0) to a corresponding 6 letter string
 // using base 26 to represent the string
 // s must be big enough to hold 6 chars plus a null (7 chars total)
__device__ __host__ void intToString(const int num, char *s)
{
   const int ones = (num) % 26;
   const int twentySix = (num / 26) % 26;
   const int twentySixSquared = (num / 26 / 26) % 26;
   const int twentySixCubed = (num / 26 / 26 / 26) % 26;
   const int twentySixFourth = (num / 26 / 26 / 26 / 26) % 26;
   const int twentySixFifth = (num / 26 / 26 / 26 / 26 / 26) % 26;
   // Store appropriate char into the string
   int i = 0;
   s[i++] = twentySixFifth + 'A';
   s[i++] = twentySixFourth + 'A';
   s[i++] = twentySixCubed + 'A';
   s[i++] = twentySixSquared + 'A';
   s[i++] = twentySix + 'A';
   s[i++] = ones + 'A';
   s[i] = '\0';
}
 
 // You may find this helpful for testing, this takes a 6 char string
 // like ABACAB and returns back the decimal number that maps to it
 // using the intToString function above
int stringToInt(char *s)
{
	const int length = strlen(s);
	int sum = 0;
	int power = 0;

	for (int i = length-1; i >= 0; i--){
		int digit = s[i] - 'A';
		sum += digit * pow(26,power);	
		power++;
	} 
	return sum;
};
 
__global__ void crack(int* result, uint32_t* md5Target){
	uint32_t hashResult1, hashResult2, hashResult3, hashResult4;
	char possibleKey[7];
	const int fifthPos = 26*26*26*26;
	const uint8_t length = 6;
	for (int i = 0; i < fifthPos; i++){
		intToString(i+blockIdx.x*26*fifthPos+threadIdx.x*fifthPos, possibleKey); 
		md5Hash((unsigned char*) possibleKey, length, &hashResult1, &hashResult2, &hashResult3, &hashResult4);
		if ((hashResult1 == md5Target[0]) && (hashResult2 == md5Target[1]) &&
			(hashResult3 == md5Target[2]) && (hashResult4 == md5Target[3])) {
			result[0] = i+blockIdx.x*26*fifthPos+threadIdx.x*fifthPos;
		}
	}
};
 // Brute force search over the space of numbers 0 - 26^6, mapped to all 6 char 
 // uppercase strings. The resulting string is hashed using md5 and compared
 // to the target hash to see if it is the same. If so, we just cracked the
 // original string that produced the md5 target.
int main(){
    // This is the md5 hash string we are trying to crack
    char md5_hash_string[] = "070d912366b1cf46a01aaf93c99f907d";
    int md5Target[4];  // The md5 hash string extracted into four integers
 
   

    // This loop extracts the md5 hash string into md5Target[0],[1],[2],[3]

    for(int i = 0; i < 4; i++){
		char tmp[16];
		strncpy(tmp, md5_hash_string + i * 8, 8);
		sscanf(tmp, "%x", &md5Target[i]);
		md5Target[i] = (md5Target[i] & 0xFF000000) >> 24 | (md5Target[i] &
                  0x00FF0000) >> 8 | (md5Target[i] & 0x0000FF00) << 8 |
                 (md5Target[i] & 0x000000FF) << 24;
    }
 
	// These variables are used to store the md5 hash for a string
	// we generate in a brute force way to test if it matches the
	// target
	int *dev_result;
	uint32_t *dev_md5Target = (uint32_t *) malloc(sizeof(uint32_t));
	int result[1];

	cudaMalloc((void **) &dev_md5Target,sizeof(uint32_t)*4);
	cudaMalloc((void **) &dev_result, sizeof(int));
	cudaMemcpy(dev_md5Target, md5Target, 4*sizeof(uint32_t),cudaMemcpyHostToDevice);

	 
	crack<<<26,26>>>(dev_result, dev_md5Target);
	cudaMemcpy(result,dev_result, sizeof(int),cudaMemcpyDeviceToHost);
	char possibleKey[7];// This is a bad duplicate.
	intToString(result[0], possibleKey);

	printf("The Key is: %s !\n",possibleKey);
	
	cudaFree(dev_md5Target);
	cudaFree(dev_result);
	// Assume we don't know the key, try brute force cracker by
	// hashing all 6 letter strings from AAAAAA to ZZZZZZ
}