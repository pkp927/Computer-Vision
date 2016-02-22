/*
 * DisplayImage.cpp
 *
 *  Created on: Jan 15, 2016
 *      Author: simi
 */

#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

String outf;

void basic(char* im, int x, int y){
	  Mat image = imread( im, IMREAD_COLOR );

	  if(!image.data ){
	  	      printf( "No image data \n" );
	  	      return;
	  }
	  namedWindow( "Display Image", WINDOW_AUTOSIZE );
	  //imshow( "Display Image", image );

	  for(int i=x; i<x+25; i++){
		  for(int j=y; j<y+25; j++){
			  image.at<Vec3b>(i,j)[0] = 255;
			  image.at<Vec3b>(i,j)[1] = 255;
			  image.at<Vec3b>(i,j)[2] = 255;
		  }
	  }

	  imshow( "Display Image", image );
	  imwrite(outf+"output_img.jpg", image );

	  waitKey(0);
}

void averaging(char* im, int w, int h){
	Mat image = imread( im, IMREAD_COLOR );
	Mat out;

	if(!image.data ){
		  printf( "No image data \n" );
		  return;
	}
	namedWindow( "Display Image", WINDOW_AUTOSIZE );
	//imshow( "Display Image", image );

	blur (image, out, Size(w,h));

	imshow( "Display Image", out );
	imwrite(outf+"avg_img.jpg", out );
	waitKey(0);
}

void median_filtering(char* im, int i){
	Mat image = imread( im, IMREAD_COLOR );
	Mat out;

	if(!image.data ){
		  printf( "No image data \n" );
		  return;
	}
	namedWindow( "Display Image", WINDOW_AUTOSIZE );
	//imshow( "Display Image", image );

	medianBlur(image, out, i);

	imshow( "Display Image", out );
	imwrite(outf+"med_fil_img.jpg", out );
	waitKey(0);
}

void histogram_equalization(char* im){
	Mat image = imread( im, IMREAD_COLOR );
	Mat out, image_gray;

	if(!image.data ){
		  printf( "No image data \n" );
		  return;
	}
	namedWindow( "Display Image", WINDOW_AUTOSIZE );
	//imshow( "Display Image", image );

	cvtColor( image, image_gray, CV_BGR2GRAY );
	equalizeHist(image_gray, out);

	imshow( "Display Image", out );
	imwrite(outf+"hist_equa_img.jpg", out );
	waitKey(0);
}

void thresholding(char* im, int t_v, int bin, int t_t){
		Mat image = imread( im, IMREAD_COLOR );
		Mat out, image_gray;

		if(!image.data ){
			  printf( "No image data \n" );
			  return;
		}

		namedWindow( "Display Image", WINDOW_AUTOSIZE );
		//imshow( "Display Image", image );

		cvtColor( image, image_gray, COLOR_RGB2GRAY );
		threshold( image_gray, out, t_v, bin, t_t );

		imshow( "Display Image", out );
		imwrite(outf+"thres_img.jpg", out );
		waitKey(0);
}

void edge_detection(char* im, int low_t){
		Mat image = imread( im, IMREAD_COLOR );
		Mat out, detected_edges, image_gray;
		int max_t = 3*low_t;
		int ratio = 3;
		int k_size = 3;

		if(!image.data ){
			printf( "No image data \n" );
			return;
		}

		namedWindow( "Display Image", WINDOW_AUTOSIZE );
		//imshow( "Display Image", image );

		out.create( image.size(), image.type() );
		cvtColor( image, image_gray, COLOR_BGR2GRAY );
		blur( image_gray, detected_edges, Size(3,3) );

		Canny( detected_edges, detected_edges, low_t, low_t*ratio, k_size );

		out = Scalar::all(0);

		image.copyTo( out, detected_edges);
		imshow( "Display Image", out );
		imwrite(outf+"edge_img.jpg", out );
		waitKey(0);
}

int main( int argc, char** argv ){

	if( argc < 2 ){
	      printf( "No image data \n" );
	      return -1;
	}
	outf = argv[3]; // output folder

	int x,y, t_t, t_v, l_t;

	// 1. Basic Image Operations
	cout<<"1. Basic Image Operations"<<endl;
	cout<<"Enter the dimensions of ROI(x,y):"<<endl;
	cin>>x; cin>>y;
	basic(argv[1],x,y);

	// 3. Basic Image Manipulation Operations
	cout<<"3. Basic Image Manipulation Operations"<<endl;
	cout<<"a. Averaging"<<endl;
	cout<<"Enter the window size(x,y):"<<endl;
	cin>>x; cin>>y;
	averaging(argv[1],x,y);
	cout<<"b. Median Filtering"<<endl;
	cout<<"Enter the window size(odd x):"<<endl;
	cin>>x;
	median_filtering(argv[1],x);
	cout<<"c. Histogram Equalization"<<endl;
	histogram_equalization(argv[1]);
	cout<<"d. Thresholding"<<endl;
	cout<<"Enter the threshold value(0-255):"<<endl;
	cin>>t_v;
	t_t = 3;
	thresholding(argv[1],t_v,255,t_t);

	// 4. Basic Image Analysis Operation
	cout<<"4. Basic Image Analysis Operation"<<endl;
	cout<<"Edge Detection"<<endl;
	l_t = 75;
	edge_detection(argv[1], l_t);

	// 2. Basic Video Operations
	cout<<"2. Basic Video Operations"<<endl;
	cout<<"Enter the dimensions of ROI(x,y):"<<endl;
	cin>>x; cin>>y;

	VideoCapture capture(argv[2]);
	if (!capture.isOpened()) {
	      printf("!!! Cannot open initialize webcam!\n" );
	      return 0;
	}

	double fps = capture.get(CV_CAP_PROP_FPS);
	int width = static_cast<int>(capture.get(CV_CAP_PROP_FRAME_WIDTH));
	int height = static_cast<int>(capture.get(CV_CAP_PROP_FRAME_HEIGHT));

    VideoWriter out(outf+"output.mp4", CV_FOURCC('m','p', '4', 'v'), fps, Size(width, height));

    if(!out.isOpened()) {
          cout <<"Error! Unable to open video file for output." << endl;
          return 0;
    }

	namedWindow("MyVideo",CV_WINDOW_AUTOSIZE);

	int xx = 0, yy = 0;
    while(1){
        Mat frame;

        bool bSuccess = capture.read(frame);

        if (!bSuccess) {
                       cout << "Cannot read the frame from video file" << endl;
                       break;
        }

        for(int i=xx; i<xx+x; i++){
        	for(int j=yy; j<yy+y; j++){
        			 frame.at<Vec3b>(i,j)[0] = 255;
        			 frame.at<Vec3b>(i,j)[1] = 255;
        			 frame.at<Vec3b>(i,j)[2] = 255;
        	 }
        }
        imshow("MyVideo", frame);
        out.write(frame);

        yy = yy+5;
        if(yy >= width){
        	xx = xx+5;
        	yy = 0;
        }
        if(xx+100 >= height){
        	xx = 0;
        }

        if(waitKey(30) == 27){
                cout << "esc key is pressed by user" << endl;
                break;
       }
    }

	return 0;
}

