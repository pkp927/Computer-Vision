
/*  Project1.cpp
 *  Author: Parneet Kaur
 */

#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <math.h>
#include <cmath>
using namespace cv;
using namespace std;

// data structure to store vertices
struct Vertex{
	double x, y, z;
	bool invalid;
};

// data structure to store edges
struct Edge{
	int v1, v2;
};

// global variables
double xa0, ya0, za0, xac, yac, zac, f;
int num_e, num_v;
Vertex* vertex;
Edge* edge;
Vertex* camera_vertex;
Vertex* image_vertex;
double** translation_matrix;
double** rotation_matrix;
double** perspective_matrix;
double r_min=0;
double r_max=255;
double i_min=-1;
double i_max=1;

/* Reads data from input file into the
 * corresponding data structures
 */
bool get_input_data(char* file_name){
	ifstream myfile (file_name);
    if (myfile.is_open()){
      string line;
	  int i=1, v=0, e=0;
	  while ( getline (myfile,line) ){
	      istringstream iss(line);
	      if(i == 1){
	    	  iss >> xa0; iss >> ya0; iss>> za0;
	    	  iss >> xac; iss >> yac; iss >> zac;
	    	  iss >> f;
	      }else if(i == 2){
	    	  iss >> num_e; iss >> num_v;
	    	  vertex = new Vertex[num_v];
	    	  edge = new Edge[num_e];
	      }else if(i > 2){
	    	  if(v<num_v){
	    		  iss >> vertex[v].x;
	    		  iss >> vertex[v].y;
	    		  iss >> vertex[v].z;
	    		  v++;
	    	  }else if(e<num_e){
	    		  iss >> edge[e].v1;
	    		  iss >> edge[e].v2;
	    		  e++;
	    	  }
	      }
	      i++;
	  }
	  myfile.close();
	  return true;
	}else{
	  cout << "Unable to open file";
	  return false;
	}
}

/* Function to compute translation matrix
 */
void calculate_translation_matrix(){
	translation_matrix = new double*[4];
	for(int i=0;i<4;i++){
		translation_matrix[i] = new double[4];
	}

	// assign values to translation matrix
	translation_matrix[0][0] = 1;
	translation_matrix[0][1] = 0;
	translation_matrix[0][2] = 0;
	translation_matrix[0][3] = -1*xa0;

	translation_matrix[1][0] = 0;
	translation_matrix[1][1] = 1;
	translation_matrix[1][2] = 0;
	translation_matrix[1][3] = -1*ya0;

	translation_matrix[2][0] = 0;
	translation_matrix[2][1] = 0;
	translation_matrix[2][2] = 1;
	translation_matrix[2][3] = -1*za0;

	translation_matrix[3][0] = 0;
	translation_matrix[3][1] = 0;
	translation_matrix[3][2] = 0;
	translation_matrix[3][3] = 1;
}

/* Function to compute rotation matrix
 */
void calculate_rotation_matrix(){
	rotation_matrix = new double*[4];
	for(int i=0;i<4;i++){
		rotation_matrix[i] = new double[4];
	}

	// calculate z vector
	double zx, zy, zz;
	double denom = sqrt(pow((xac-xa0),2)+pow((yac-ya0),2)+pow((zac-za0),2));
	zx = (xac-xa0)/denom;
	zy = (yac-ya0)/denom;
	zz = (zac-za0)/denom;
	cout<<zx<<" "<<zy<<" "<<zz<<endl;

	// calculate x vector
	double xx, xy, xz;
	if(zx||zy){
		xx = abs(zy)/sqrt(pow(zx,2)+pow(zy,2));
		xy = -1*abs(zx)/sqrt(pow(zx,2)+pow(zy,2));
	}else{
		xx = 1;
		xy = 0;
	}
	xz = 0;
	cout<<xx<<" "<<xy<<" "<<xz<<endl;

	// calculate y vector
	double yx, yy, yz;
	yx = zy*xz - xy*zz;
	yy = -1*(zx*xz-xx*zz);
	yz = zx*xy - xx*zy;
	cout<<yx<<" "<<yy<<" "<<yz<<endl;

	// assign values to rotation matrix
	rotation_matrix[0][0] = xx;
	rotation_matrix[0][1] = xy;
	rotation_matrix[0][2] = xz;
	rotation_matrix[0][3] = 0;

	rotation_matrix[1][0] = yx;
	rotation_matrix[1][1] = yy;
	rotation_matrix[1][2] = yz;
	rotation_matrix[1][3] = 0;

	rotation_matrix[2][0] = zx;
	rotation_matrix[2][1] = zy;
	rotation_matrix[2][2] = zz;
	rotation_matrix[2][3] = 0;

	rotation_matrix[3][0] = 0;
	rotation_matrix[3][1] = 0;
	rotation_matrix[3][2] = 0;
	rotation_matrix[3][3] = 1;

}

/* Function to compute perspective matrix
 */
void calculate_perspective_matrix(){
	perspective_matrix = new double*[4];
	for(int i=0;i<4;i++){
		perspective_matrix[i] = new double[4];
	}

	// assign values to perspective matrix
	perspective_matrix[0][0] = 1;
	perspective_matrix[0][1] = 0;
	perspective_matrix[0][2] = 0;
	perspective_matrix[0][3] = 0;

	perspective_matrix[1][0] = 0;
	perspective_matrix[1][1] = 1;
	perspective_matrix[1][2] = 0;
	perspective_matrix[1][3] = 0;

	perspective_matrix[2][0] = 0;
	perspective_matrix[2][1] = 0;
	perspective_matrix[2][2] = 1;
	perspective_matrix[2][3] = 0;

	perspective_matrix[3][0] = 0;
	perspective_matrix[3][1] = 0;
	perspective_matrix[3][2] = -1/double(f);
	perspective_matrix[3][3] = 1;
}

/* Function to compute multiplication of two matrices
 */
double** matrix_multiply(double** m1, double** m2, int r, int c, int m){
	double** result = new double*[r];
	double sum;
	for(int i=0;i<r;i++){
		result[i] = new double[c];
		for(int j=0;j<c;j++){
			sum = 0;
			for(int k=0;k<m;k++){
				sum = sum + m1[i][k]*m2[k][j];
			}
			result[i][j] = sum;
		}
	}
	return result;
}

/* Function to compute image coordinates from given
 * object coordinates and transformation matrices
 */
void compute_image(){
	image_vertex = new Vertex[num_v];
	camera_vertex = new Vertex[num_v];

	// compute the translation matrix = P X R X T
	//double** trans = matrix_multiply(perspective_matrix,matrix_multiply( rotation_matrix, translation_matrix,4,4,4),4,4,4);
	double** trans = matrix_multiply(rotation_matrix, translation_matrix, 4,4,4);

	// compute image points from translation matrix and object points
	for(int i=0; i<num_v; i++){
		double** temp1 = new double*[4];
		for(int j=0;j<4;j++){
			temp1[j] = new double[1];
		}
		temp1[0][0] = vertex[i].x;
		temp1[1][0] = vertex[i].y;
		temp1[2][0] = vertex[i].z;
		temp1[3][0] = 1;

		// compute camera coordinate vertices
		double** temp2 = matrix_multiply(trans, temp1, 4, 1, 4);
		camera_vertex[i].x = temp2[0][0]/temp2[3][0];
		camera_vertex[i].y = temp2[1][0]/temp2[3][0];
		camera_vertex[i].z = temp2[2][0]/temp2[3][0];

		// compute image vertices
		double** temp3 = matrix_multiply(perspective_matrix, temp2, 4, 1, 4);
		//cout<<temp1[0][0]<<" "<<temp1[1][0]<<" "<<temp1[2][0]<<" "<<temp1[3][0]<<endl;
		//cout<<temp2[0][0]<<" "<<temp2[1][0]<<" "<<temp2[2][0]<<" "<<temp2[3][0]<<endl;
		//cout<<temp3[0][0]<<" "<<temp3[1][0]<<" "<<temp3[2][0]<<" "<<temp3[3][0]<<endl;

		if(camera_vertex[i].z < f){       // check for invalid vertices
			image_vertex[i].invalid = true;
			cout<<"invalid";
		}else{
			image_vertex[i].invalid = false;
		}

		image_vertex[i].x = temp3[0][0]/temp3[3][0];
		image_vertex[i].y = temp3[1][0]/temp3[3][0];
		image_vertex[i].z = temp3[2][0]/temp3[3][0];
		cout<<image_vertex[i].x<<" "<<image_vertex[i].y<<" "<<image_vertex[i].z<<endl;
	}

	// testing
	//double dist = sqrt(pow(image_vertex[2].x - image_vertex[0].x,2) +
			//pow(image_vertex[2].y - image_vertex[0].y,2) +
			//pow(image_vertex[2].z - image_vertex[0].z,2));
	//cout<<dist<<endl;
}

/* Function to scale the image to pixels
 */
void rescale_image(double r_min, double r_max, double min, double max){
	for(int i=0; i<num_v; i++){
		image_vertex[i].x = r_min + ((r_max - r_min)*(image_vertex[i].x - min))/(max-min);
		image_vertex[i].y = r_min + ((r_max - r_min)*(image_vertex[i].y - min))/(max-min);
		image_vertex[i].z = r_min + ((r_max - r_min)*(image_vertex[i].z - min))/(max-min);
		//cout<<image_vertex[i].x<<" "<<image_vertex[i].y<<" "<<image_vertex[i].z<<endl;
		if(!isfinite(image_vertex[i].x) && !isfinite(image_vertex[i].y)){
			if(isnan(image_vertex[i].x)){
				image_vertex[i].x = -1;
			}
			if(isnan(image_vertex[i].y)){
				image_vertex[i].y = -1;
			}
			if(image_vertex[i].x == INFINITY){
				image_vertex[i].x =-1;
			}
			if(image_vertex[i].x == -INFINITY){
				image_vertex[i].x =256;
			}
			if(image_vertex[i].y == INFINITY){
				image_vertex[i].y =-1;
			}
			if(image_vertex[i].y == -INFINITY){
				image_vertex[i].y =256;
			}
		}
	}
}

/* Function to create image*/
void create_image(Mat image){
	int thickness = 1;
	int lineType = 8;
	for(int i=0; i<num_e; i++){
		if(!image_vertex[edge[i].v1-1].invalid && !image_vertex[edge[i].v2-1].invalid){
			Point p1(image_vertex[edge[i].v1-1].x,image_vertex[edge[i].v1-1].y);
			Point p2(image_vertex[edge[i].v2-1].x, image_vertex[edge[i].v2-1].y);
			line( image,p1,p2,Scalar( 255, 255, 255),thickness,lineType );
		}else if(image_vertex[edge[i].v1-1].invalid && !image_vertex[edge[i].v2-1].invalid){
			// calculate line equation for these points
			double x = camera_vertex[edge[i].v1-1].x - camera_vertex[edge[i].v2-1].x;
			double y = camera_vertex[edge[i].v1-1].y - camera_vertex[edge[i].v2-1].y;
			double z = camera_vertex[edge[i].v1-1].z - camera_vertex[edge[i].v2-1].z;
			// calculate intersection point
			double t = (f - camera_vertex[edge[i].v1-1].z)/z;
			double ix = t*x + camera_vertex[edge[i].v1-1].x;
			double iy = t*y + camera_vertex[edge[i].v1-1].y;
			double iz = f;
			//cout<<"new"<<ix<<" "<<iy<<" "<<iz<<endl;
			// compute new image vertex
			double** temp1 = new double*[4];
			for(int j=0;j<4;j++){
				temp1[j] = new double[1];
			}
			temp1[0][0] = ix;
			temp1[1][0] = iy;
			temp1[2][0] = iz;
			temp1[3][0] = 1;
			double** temp2 = matrix_multiply(perspective_matrix, temp1, 4, 1, 4);
			image_vertex[edge[i].v1-1].x = temp2[0][0]/temp2[3][0];
			image_vertex[edge[i].v1-1].y = temp2[1][0]/temp2[3][0];
			image_vertex[edge[i].v1-1].z = temp2[2][0]/temp2[3][0];
			// rescale it
			image_vertex[edge[i].v1-1].x = r_min + ((r_max - r_min)*(image_vertex[edge[i].v1-1].x - i_min))/(i_max-i_min);
			image_vertex[edge[i].v1-1].y = r_min + ((r_max - r_min)*(image_vertex[edge[i].v1-1].y - i_min))/(i_max-i_min);
			image_vertex[edge[i].v1-1].z = r_min + ((r_max - r_min)*(image_vertex[edge[i].v1-1].z - i_min))/(i_max-i_min);

			//cout<<image_vertex[edge[i].v2-1].x<<" "<<image_vertex[edge[i].v2-1].y<<" "<<image_vertex[edge[i].v2-1].z<<endl;
			if(isinf(image_vertex[edge[i].v1-1].x) && isinf(image_vertex[edge[i].v1-1].y)){
				if(isnan(image_vertex[edge[i].v1-1].x)){
					image_vertex[edge[i].v1-1].x = -1;
				}
				if(isnan(image_vertex[edge[i].v1-1].y)){
					image_vertex[edge[i].v1-1].y = -1;
				}
				if(image_vertex[edge[i].v1-1].x == INFINITY){
					image_vertex[edge[i].v1-1].x =-1;
				}
				if(image_vertex[edge[i].v1-1].x == -INFINITY){
					image_vertex[edge[i].v1-1].x =256;
				}
				if(image_vertex[edge[i].v1-1].y == INFINITY){
					image_vertex[edge[i].v1-1].y =-1;
				}
				if(image_vertex[edge[i].v1-1].y == -INFINITY){
					image_vertex[edge[i].v1-1].y =256;
				}
			}
			cout<<"new"<<image_vertex[edge[i].v1-1].x<<" "<<image_vertex[edge[i].v1-1].y<<" "<<image_vertex[edge[i].v1-1].z<<endl;

			// draw the line as image
			Point p1(image_vertex[edge[i].v1-1].x,image_vertex[edge[i].v1-1].y);
			Point p2(image_vertex[edge[i].v2-1].x, image_vertex[edge[i].v2-1].y);
			line( image,p1,p2,Scalar( 255, 255, 255),thickness,lineType );
		}else if(!image_vertex[edge[i].v1-1].invalid && image_vertex[edge[i].v2-1].invalid){
			// calculate line equation for these points
			double x = camera_vertex[edge[i].v1-1].x - camera_vertex[edge[i].v2-1].x;
			double y = camera_vertex[edge[i].v1-1].y - camera_vertex[edge[i].v2-1].y;
			double z = camera_vertex[edge[i].v1-1].z - camera_vertex[edge[i].v2-1].z;
			// calculate intersection point
			double t = (f - camera_vertex[edge[i].v1-1].z)/z;
			double ix = t*x + camera_vertex[edge[i].v1-1].x;
			double iy = t*y + camera_vertex[edge[i].v1-1].y;
			double iz = f;
			//cout<<"new"<<ix<<" "<<iy<<" "<<iz<<endl;
			// compute new image vertex
			double** temp1 = new double*[4];
			for(int j=0;j<4;j++){
				temp1[j] = new double[1];
			}
			temp1[0][0] = ix;
			temp1[1][0] = iy;
			temp1[2][0] = iz;
			temp1[3][0] = 1;
			double** temp2 = matrix_multiply(perspective_matrix, temp1, 4, 1, 4);
			image_vertex[edge[i].v2-1].x = temp2[0][0]/temp2[3][0];
			image_vertex[edge[i].v2-1].y = temp2[1][0]/temp2[3][0];
			image_vertex[edge[i].v2-1].z = temp2[2][0]/temp2[3][0];
			// rescale it
			image_vertex[edge[i].v2-1].x = r_min + ((r_max - r_min)*(image_vertex[edge[i].v2-1].x - i_min))/(i_max-i_min);
			image_vertex[edge[i].v2-1].y = r_min + ((r_max - r_min)*(image_vertex[edge[i].v2-1].y - i_min))/(i_max-i_min);
			image_vertex[edge[i].v2-1].z = r_min + ((r_max - r_min)*(image_vertex[edge[i].v2-1].z - i_min))/(i_max-i_min);

			//cout<<image_vertex[edge[i].v2-1].x<<" "<<image_vertex[edge[i].v2-1].y<<" "<<image_vertex[edge[i].v2-1].z<<endl;
			if(isinf(image_vertex[edge[i].v2-1].x) && isinf(image_vertex[edge[i].v2-1].y)){
				if(isnan(image_vertex[edge[i].v2-1].x)){
					image_vertex[edge[i].v2-1].x = -1;
				}
				if(isnan(image_vertex[edge[i].v2-1].y)){
					image_vertex[edge[i].v2-1].y = -1;
				}
				if(image_vertex[edge[i].v2-1].x == INFINITY || image_vertex[edge[i].v2-1].x == NAN){
					image_vertex[edge[i].v2-1].x =-1;
				}
				if(image_vertex[edge[i].v2-1].x == -INFINITY|| image_vertex[edge[i].v2-1].x == -NAN){
					image_vertex[edge[i].v2-1].x =256;
				}
				if(image_vertex[edge[i].v2-1].y == INFINITY|| image_vertex[edge[i].v2-1].y == NAN){
					image_vertex[edge[i].v2-1].y =-1;
				}
				if(image_vertex[edge[i].v2-1].y == -INFINITY|| image_vertex[edge[i].v2-1].y == -NAN){
					image_vertex[edge[i].v2-1].y =256;
				}
			}
			cout<<"new"<<image_vertex[edge[i].v2-1].x<<" "<<image_vertex[edge[i].v2-1].y<<" "<<image_vertex[edge[i].v2-1].z<<endl;

			// draw the line as image
			Point p1(image_vertex[edge[i].v1-1].x,image_vertex[edge[i].v1-1].y);
			Point p2(image_vertex[edge[i].v2-1].x, image_vertex[edge[i].v2-1].y);
			line( image,p1,p2,Scalar( 255, 255, 255),thickness,lineType );
		}
	}
}

int main( int argc, char** argv ){

     if( argc < 2 ){
	 printf( "No data file \n" );
	 return -1;
     }
     // get input from file
     if(!get_input_data(argv[1])) return 0;

    // print the input vertices
    for(int i=0;i<num_v;i++){
    		//cout<<vertex[i].x<<" "<<vertex[i].y<<" "<<vertex[i].z<<endl;
    }
    // print the input edges
    for(int i=0;i<num_e;i++){
    	//cout<<edge[i].v1<<" "<<edge[i].v2<<endl;
    }

    // calculate transformation matricies
    calculate_translation_matrix();
    calculate_rotation_matrix();
    calculate_perspective_matrix();

    // compute the image
    compute_image();
    rescale_image(0,255,-1,1);

    // create image and display and store it
    Mat image = Mat::zeros( 256, 256, CV_8UC3 );
    create_image(image);
    namedWindow( "Display Image", WINDOW_AUTOSIZE );
    imshow( "Display Image", image );
    if(argc>2){
    	String outf = argv[2];
    	imwrite(outf+"out_img.jpg", image );
    }
    waitKey(0);
}
