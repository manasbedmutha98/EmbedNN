 //program to turn on the RGB LED on the onboard in an alternating fashion.

//ptb 18 red
//ptb 19 green
//ptd 1 blue
/*
28 x 28 x 1 -> 24 x 24 x 8 -> 12 x 12 x 8 -> 10 x 10 x 3 -> 5 x 5 x 3 -> 10

*/

#include <stdio.h>
#include <math.h>

#define conv1_filters 8
#define conv2_filters 3

const int conv1_size = 5;
const int conv2_size = 3;

const int pool1_size = 2;
const int pool2_size = 2;

const int conv1_out = 28 + 1 - conv1_filters;   //24
const int conv2_out = (conv1_out/pool1_size) - conv2_filters;  //10

//const inputs
const float convLayer1_Filter[conv1_filters][conv1_size*conv1_size] = { {-0.043727, -0.319542, -0.932302, -0.958231, -0.760573, 0.319465, 0.740485, -0.0628786, -0.77444, -1.02028, 0.00786517, 1.32189, 1.36919, 0.297155, -0.80239, -0.534547, 0.483615, 1.18557, 0.784956, -0.335773, -0.7561, -0.625647, 0.0181124, 0.472003, 0.0574038}, {0.385656, 0.270036, 0.227005, 0.389672, 0.833664, 0.38794, 0.740377, 0.808299, 0.786636, 0.469562, -0.476432, 0.280069, 0.380469, 0.236093, -0.455962, -1.25828, -0.733766, -0.639667, -0.595802, -0.915336, -0.977895, -1.08739, -1.26118, -0.997041, -0.843509}, {-0.881902, -0.858412, -0.813534, -0.453437, 0.315307, -0.722251, -0.660499, -0.364824, 0.455259, 0.961599, -0.880827, -0.696499, -0.085748, 0.948995, 0.784802, -0.996555, -0.599762, 0.237379, 0.923809, 0.544108, -1.10954, -0.762359, -0.206164, 0.449237, 0.39458}};;

const float convLayer2_Filter[conv2_filters][conv1_filters][conv2_size*conv2_size];

const float fcLayerWeights[(conv2_out/pool2_size) * (conv2_out/pool2_size) * conv2_filters * 10];

float input[784] = { 0.203922, 0.0130719, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.00784314, 0.0928105, 0.0993464, 0.0300654, 0.00130719, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0941176, 0.465359, 0.509804, 0.2, 0.0248366, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.10719, 0.534641, 0.598693, 0.236601, 0.0287582, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.180392, 0.704575, 0.811765, 0.351634, 0.0496732, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.188235, 0.718954, 0.822222, 0.356863, 0.0509804, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.222222, 0.754248, 0.752941, 0.275817, 0.0300654, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.00261438, 0.262745, 0.786928, 0.699346, 0.219608, 0.0169935, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0287582, 0.341176, 0.806536, 0.682353, 0.214379, 0.0169935, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0313726, 0.332026, 0.783007, 0.764706, 0.299346, 0.0366013, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0313726, 0.326797, 0.777778, 0.776471, 0.308497, 0.0379085, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0287582, 0.30719, 0.766013, 0.796078, 0.324183, 0.0405229, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.00261438, 0.166013, 0.654902, 0.861438, 0.433987, 0.0732026, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.075817, 0.513726, 0.88366, 0.534641, 0.105882, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0104575, 0.339869, 0.828758, 0.594771, 0.132026, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.054902, 0.424837, 0.844444, 0.568627, 0.12549, 0.00130719, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0366013, 0.392157, 0.830065, 0.648366, 0.192157, 0.0143791, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.075817, 0.500654, 0.856209, 0.613072, 0.179085, 0.0143791, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.104575, 0.577778, 0.890196, 0.52549, 0.108497, 0.00130719, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.138562, 0.654902, 0.94902, 0.581699, 0.126797, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.101961, 0.563399, 0.878431, 0.499346, 0.0941176, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0339869, 0.312418, 0.709804, 0.411765, 0.0653595, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.00261438, 0.163399, 0.67451, 0.437909, 0.0705882, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0130719, 0.139869, 0.0797386, 0.00653595, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};


//tempout
float convLayer1_Out[conv1_filters * conv1_out * conv1_out];
float poolLayer1_Out[conv1_filters * conv1_out * conv1_out / pool1_size * pool1_size];
	
float convLayer2_Out[conv2_filters * conv2_out * conv2_out];
float poolLayer2_Out[conv2_filters * conv2_out * conv2_out / pool2_size * pool2_size];
	
float out[10];

//other stuff
struct point_t {
	int x, y, z;
};

int getIndex(int _x, int _y, int _z, int size) {
	return (_z * (size * size) + _y * (size) + _x);
}
int getIndexTwo(int _x, int _y, int _z) {
	return _z * (12 * 12 * conv1_filters * 10)
			+ _y * (12 * 12 * conv1_filters) + _x;
}

struct point_t map_to_input(struct point_t out, int z, int stride) {
	out.x *= stride;
	out.y *= stride;
	out.z = z;
	return out;
}

//

void convLayer1() {
	for (int filter = 0; filter<conv1_filters; filter++) {
		for (int x = 0; x < conv1_out; x++) {
			for (int y = 0; y < conv1_out; y++) {
				struct point_t temp;
				temp.x = x;
				temp.y = y;
				temp.z = 0;
				struct point_t mapped = map_to_input(temp, 0, 1);
				float sum = 0;
				for (int i = 0; i < conv1_size; i++)
					for (int j = 0; j < conv1_size; j++)
						for (int z = 0; z < 1; z++) { // Can't z = 0 directly?
							float f = convLayer1_Filter[filter][getIndex(i, j, z,
									conv1_size)];
							float v = input[getIndex(mapped.x + i,
									mapped.y + j, z, 28)];
							sum += f * v;
						}
						
				
					// Add ReLU inside	
				if (sum>0){
					convLayer1_Out[getIndex(x, y, filter, conv1_out)] = sum;
				}
				else{
					convLayer1_Out[getIndex(x, y, filter, conv1_out)] = 0;
				}

				
				//convLayer1_Out[getIndex(x, y, filter, conv1_out)] = sum;
			}
		}
	}
}

void convLayer2() {
	for (int filter = 0; filter < conv2_filters; filter++) {
		for (int x = 0; x < conv2_out; x++) {
			for (int y = 0; y < conv2_out; y++) {
				for(int d=0; d < conv1_filters; d++) { // Depth
					struct point_t temp;
					temp.x = x;
					temp.y = y;
					temp.z = 0;
					struct point_t mapped = map_to_input(temp, 0, 1);
					float sum = 0;
					for (int i = 0; i < conv2_size; i++)
						for (int j = 0; j < conv2_size; j++)
							for (int z = 0; z < 1; z++) { // Can't z = 0 directly?
								float f = convLayer2_Filter[filter][d][getIndex(i, j, z,
										conv2_size)];
								float v = poolLayer1_Out[getIndex(mapped.x + i,
										mapped.y + j, z, conv1_out/pool1_size)];
								sum += f * v;
							}
						
						
					// Add ReLU inside	
					if (sum>0){
						convLayer2_Out[getIndex(x, y, filter, conv2_out)] = sum;
					}
					else{
						convLayer2_Out[getIndex(x, y, filter, conv2_out)] = 0;
					}
				
				}
			}
		}
	}
}




//void reluLayer() {
//	for (int i = 0; i < 24; i++)
//		for (int j = 0; j < 24; j++)
//			for (int z = 0; z < conv1_filters; z++) {
//				float v = convLayerOut[getIndex(i, j, z, 24)];
//				if (v < 0)
//					v = 0;
//				convLayerOut[getIndex(i, j, z, 24)] = v;
//			}

//}

void poolLayer1() {
	for (int x = 0; x < conv1_out/pool1_size; x++) {
		for (int y = 0; y < conv1_out/pool1_size; y++) {
			for (int z = 0; z < conv1_filters; z++) {
				struct point_t temp;
				temp.x = x;
				temp.y = y;
				temp.z = 0;
				struct point_t mapped = map_to_input(temp, 0, 2);
				float mval = convLayer1_Out[getIndex(mapped.x, mapped.y, z, conv1_out)];
				for (int i = 0; i < 2; i++)
					for (int j = 0; j < 2; j++) {
						float v = convLayer1_Out[getIndex(mapped.x + i,
								mapped.y + j, z, 24)];
						if (v > mval)
							mval = v;
					}
				poolLayer1_Out[getIndex(x, y, z, 12)] = mval;
			}
		}
	}
}


void poolLayer2() {
	for (int x = 0; x < conv2_out/pool2_size; x++) {
		for (int y = 0; y < conv2_out/pool2_size; y++) {
			for (int z = 0; z < conv2_filters; z++) {
				struct point_t temp;
				temp.x = x;
				temp.y = y;
				temp.z = 0;
				struct point_t mapped = map_to_input(temp, 0, 2);
				float mval = convLayer2_Out[getIndex(mapped.x, mapped.y, z, conv2_out)];
				for (int i = 0; i < 2; i++)
					for (int j = 0; j < 2; j++) {
						float v = convLayer2_Out[getIndex(mapped.x + i,
								mapped.y + j, z, 24)];
						if (v > mval)
							mval = v;
					}
				poolLayer2_Out[getIndex(x, y, z, 12)] = mval;
			}
		}
	}
}



void fcLayer() {
	for (int n = 0; n < 10; n++) {
		float inputv = 0;

		for (int i = 0; i < 5; i++) // 5 = conv2_out / pool2_size
			for (int j = 0; j < 5; j++) // 5 = conv2_out / pool2_size
				for (int z = 0; z < conv2_filters; z++) {
					int m = z * (5 * 5) + j * (5) + i;
					inputv += poolLayer2_Out[getIndex(i, j, z, 12)]
							* fcLayerWeights[getIndexTwo(m, n, 0)];
				}

		out[n] = 1.0f / (1.0f + exp(-inputv));
	}

}
unsigned runner() {

	convLayer1();
	//reluLayer();
	poolLayer1();
	
	convLayer2();
	poolLayer2();
	
	fcLayer();
	
	unsigned index;
	int max =0;
	int temp =0;
	for(unsigned i =0;i<10;i++){
		temp = out[i]* 100;
		if(max<temp){
			index = i;
			max = temp;
		}
	}
	return index;

}


int main()
{
	unsigned x, i;
	x = runner();
	i = 0;
}
