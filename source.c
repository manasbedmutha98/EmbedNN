/*
 * source.c
 *
 *  Created on: 28-Mar-2019
 *      Author: glint
 */
#include<stdio.h>
#include <math.h>
#define num_filters 8

float input[1][28][28];
float wmap1[num_filters][5][5];
float wmap2[10][1152];
float bmap1[num_filters];
float bmap2[10];
float fmap1[num_filters][24][24];
float fmap2[num_filters][12][12];
float dense[num_filters * 144][1];
float output[10];
float matRes[10][1];

void Conv2D() {
	for (int z = 0; z < num_filters; z++) {
		for (int channel = 0; channel < 1; channel++) {
			for (int x = 0; x < 24; x++) {
				for (int y = 0; y < 24; y++) {
					//superimpose the filter. multiply them and add all of them together.
					float sum = 0;
					for (int xf = 0; xf < 5; xf++) {
						for (int yf = 0; yf < 5; yf++) {
							sum = sum
									+ input[channel][x + xf][y + yf]
											* wmap1[z][xf][yf];
						}
					}
					sum = sum + bmap1[z];
					if (sum >= 0) {
						fmap1[z][x][y] = sum;
					} else {
						fmap1[z][x][y] = 0;
					}

				}
			}
		}
	}
}

void MaxPooling() {
	float max = 0;
	for (int imz = 0; imz < num_filters; imz++) {
		for (int imx = 0; imx < 24; imx = imx + 2) {
			for (int imy = 0; imy < 24; imy = imy + 2) {
				max = 0;
				for (int i = 0; i < 2; i++) {
					for (int j = 0; j < 2; j++) {
						if (fmap1[imz][imx + i][imy + j] > max)
							max = fmap1[imz][imx + i][imy + j];
					}
				}
				fmap2[imz][imx / 2][imy / 2] = max;
			}
		}
	}
}

void Flatten() {
	for (int imz = 0; imz < num_filters; imz++) {
		for (int imy = 0; imy < 12; imy = imy + 1) {
			for (int imx = 0; imx < 12; imx = imx + 1) {
				dense[(12 * 12 * imz) + (12 * imx) + imy][0] =
						fmap2[imz][imx][imy];
			}
		}
	}
}

void Dense() {
	float inner = 0;
	for (int i = 0; i < 10; i++) {
		for (int j = 0; j < 1; j++) {
			for (int k = 0; k < 1152; k++) {
				inner = inner + wmap2[i][k] * dense[k][j];
			}
			matRes[i][j] = inner;
			inner = 0;
		}
	}

	for (int i = 0; i < 10; i++) {
		for (int j = 0; j < 1; j++) {
			matRes[i][j] = matRes[i][j] + bmap2[i];
		}
	}
	float sum=0;
	for (int i = 0; i < 10; i++) {
		for (int j = 0; j < 1; j++) {
			float ex = exp(matRes[i][j]);
			output[i] = ex;
			sum = sum + ex;
		}
	}

	for (int i = 0; i < 10; i++) {
		output[i] = output[i] / sum;
	}

}

int main() {
	Conv2D();
	MaxPooling();
	Flatten();
	Dense();

	for (int i = 0; i < 10; i++) {
		printf("[%d] %f\n", i, output[i]);
	}

}
