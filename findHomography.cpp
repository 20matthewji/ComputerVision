/**
 * @file SIFT_Homography
 * @brief SIFT detector + descriptor + FLANN Matcher + FindHomography
 */

#include <stdio.h>
#include <iostream>
using namespace std;

#ifndef HAVE_OPENCV_NONFREE

int main(int, char**)
{
	printf("The sample requires nonfree module that is not available in your OpenCV distribution.\n");
	return -1;
}

#else

#include "opencv2/opencv_modules.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/nonfree/features2d.hpp"
#include "opencv2/calib3d/calib3d.hpp"

using namespace cv;

void readme();

/**
 * @function main
 * @brief Main function
 */
int main( int argc, char** argv )
{
	static char *file1=	"/Users/matthewjin/Downloads/Extracurricular/SRC/DroneImageTests/DroneFlightImages1/DroneImageTest1_9.jpg";
	static char *file2=	"/Users/matthewjin/Documents/workspace/SRC/House/Dumas.png";
	char *n_argv[] = {"findHomography", "/Users/Matthew/Documents/Vision/DroneImages/DroneImageTest1_9.jpg",
			"/Users/matthewjin/Documents/workspace/SRC/House/Dumas.png"};
	argv=n_argv;
	argc = 3;

	if( argc != 3 )
	{ readme(); return -1; }

	Mat img_scene = imread(argv[1], CV_LOAD_IMAGE_COLOR );
	Mat img_object = imread(argv[2], IMREAD_COLOR );
/*
	namedWindow( "Display window", WINDOW_AUTOSIZE );	// Create a window for display.
	cv::resizeWindow("Display window", 1080, 1080);
	imshow("Display window", img_scene);
*/
	std::cout << "imgobject: " << img_object.cols << " x " << img_object.rows << "\n";
	std::cout << "imgscene: " << img_scene.cols << " x " << img_scene.rows << "\n";

	if( !img_object.data || !img_scene.data )
	{ printf(" --(!) Error reading images \n"); return -1; }

	//-- Step 1: Detect the keypoints using SIFT Detector

	SiftFeatureDetector detector();

	std::vector<KeyPoint> keypoints_object, keypoints_scene;

	detector.detect( img_object, keypoints_object );
	detector.detect( img_scene, keypoints_scene );
	std::cout << "k_objsize: " << keypoints_object.size() << "\tk_scenesize: " << keypoints_scene.size() << "\n";

	//-- Step 2: Calculate descriptors (feature vectors)
	SiftDescriptorExtractor extractor;

	Mat descriptors_object, descriptors_scene;

	extractor.compute( img_object, keypoints_object, descriptors_object );
	extractor.compute( img_scene, keypoints_scene, descriptors_scene );

	detector.detect(img_scene, keypoints_scene);
	extractor.compute (img_scene, keypoints_scene, descriptors_scene);

	//-- Step 3: Matching descriptor vectors using FLANN matcher
	FlannBasedMatcher matcher;
	std::vector< DMatch > matches;
	matcher.match( descriptors_object, descriptors_scene, matches );

	double max_dist = 0; double min_dist = 100;

	//-- Quick calculation of max and min distances between keypoints
	for( int i = 0; i < descriptors_object.rows; i++ )
	{ double dist = matches[i].distance;
	if( dist < min_dist ) min_dist = dist;
	if( dist > max_dist ) max_dist = dist;
	}

	printf("-- Max dist : %f \n", max_dist );
	printf("-- Min dist : %f \n", min_dist );

	//-- Draw only "good" matches (i.e. whose distance is less than 3*min_dist )
	std::vector< DMatch > good_matches;

	for( int i = 0; i < descriptors_object.rows; i++ )
	{ if( matches[i].distance < 3*min_dist )
	{ good_matches.push_back( matches[i]); }
	}

	Mat img_matches;
	drawMatches( img_object, keypoints_object, img_scene, keypoints_scene,
			good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
			vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );


	//-- Localize the object from img_1 in img_2
	std::vector<Point2f> obj;
	std::vector<Point2f> scene;

	for( size_t i = 0; i < good_matches.size(); i++ )
	{
		//-- Get the keypoints from the good matches
		obj.push_back( keypoints_object[ good_matches[i].queryIdx ].pt );
		scene.push_back( keypoints_scene[ good_matches[i].trainIdx ].pt );
	}

	std::cout << "objsize: " << obj.size() << "\tscenesize: " << scene.size() << "\n";
	Mat H = findHomography( obj, scene, CV_RANSAC );

	//-- Get the corners from the image_1 ( the object to be "detected" )
	std::vector<Point2f> obj_corners(4);
	obj_corners[0] = Point(0,0); obj_corners[1] = Point( img_object.cols, 0 );
	obj_corners[2] = Point( img_object.cols, img_object.rows ); obj_corners[3] = Point( 0, img_object.rows );
	std::vector<Point2f> scene_corners(4);

	perspectiveTransform( obj_corners, scene_corners, H);


	//-- Draw lines between the corners (the mapped object in the scene - image_2 )
	Point2f offset( (float)img_object.cols, 0);
	line( img_matches, scene_corners[0] + offset, scene_corners[1] + offset, Scalar(0, 255, 0), 4 );
	line( img_matches, scene_corners[1] + offset, scene_corners[2] + offset, Scalar( 0, 255, 0), 4 );
	line( img_matches, scene_corners[2] + offset, scene_corners[3] + offset, Scalar( 0, 255, 0), 4 );
	line( img_matches, scene_corners[3] + offset, scene_corners[0] + offset, Scalar( 0, 255, 0), 4 );

	//-- Show detected matches
	imshow( "Good Matches & Object detection", img_matches );

	waitKey(0);

	return 0;
}

/**
 * @function readme
 */
void readme()
{ printf(" Usage: ./SIFT_Homography <img1> <img2>\n"); }

#endif
