/*

Convolution Layer

Inputs:

Input Feature Map
Filters
Stride
Padding

Output:
Feature map

*/
float input[]  = {{1.0,4.0,3.5},{5.0,2.65,7.95},{4.1,8.23,7.99}};
float filter[] = {{5.0,3.1},{9.9,8.0},{11.5,9.5}};
float output[size_i[2]-size_f[2]][size_i[1]-size_f[1]][size_i[0]-size_f[0]];

float convlayer(){
	for i=0;i<size_i[0]-size_f[0];i++
	{
		for j<0;j<size_i[1]-size_f[1];j++
		{
			for k<0;k<size_i[2]-size_f[2];k++
			{
				for l=0;l<size_f[0];i++
				{
					for m=0;l<size_f[1];i++
					{
						for n=0;l<size_f[1];i++
							ouput[i][j][k] = input[n+k][m+j][i+l]*filter[n][m][l];
					}
				}
			}
		}
	}

}

