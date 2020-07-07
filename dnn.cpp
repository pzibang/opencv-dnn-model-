#include <fstream>
#include <sstream>

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

using namespace cv;
using namespace dnn;

std::vector<std::string> classes;

int main(int argc, char** argv)
{
	float scale{ 1.0 };
	Scalar mean{ 0, 0, 0 };
	bool swapRB{ false };
	int inpWidth = 416;
	int inpHeight = 416;
	String model = "./deploy.prototxt";
	String config = "./squeezenet_v1.1.caffemodel";
	String framework = "";
	int backendId = cv::dnn::DNN_BACKEND_OPENCV;
	int targetId = cv::dnn::DNN_TARGET_OPENCL;

	String classesFile = "./classification_classes_ILSVRC2012.txt";

	// Open file with classes names.
	if (!classesFile.empty()) {
		const std::string& file = classesFile;
		std::ifstream ifs(file.c_str());
		if (!ifs.is_open())
			CV_Error(Error::StsError, "File " + file + " not found");
		std::string line;
		while (std::getline(ifs, line)) {
			classes.push_back(line);
		}
	}

	CV_Assert(!model.empty());

	//! [Read and initialize network]
	Net net = readNet(model, config, framework);
	//Net net = readNetFromCaffe(model, config); //明确框架类型
	net.setPreferableBackend(backendId);
	net.setPreferableTarget(targetId);
	//! [Read and initialize network]

	// Create a window
	static const std::string kWinName = "Deep learning image classification in OpenCV";
	namedWindow(kWinName, WINDOW_NORMAL);

	//! [Open a video file or an image file or a camera stream]
	VideoCapture cap;
	cap.open(0);
	//! [Open a video file or an image file or a camera stream]

	// Process frames.
	Mat frame, blob;
	while (waitKey(1) < 0) {
		cap >> frame;
		if (frame.empty()) {
			waitKey();
			break;
		}

		//! [Create a 4D blob from a frame]
		blobFromImage(frame, blob, scale, Size(inpWidth, inpHeight), mean, swapRB, false);
		//! [Create a 4D blob from a frame]

		//! [Set input blob]
		net.setInput(blob);
		//! [Set input blob]
		//! [Make forward pass]
		Mat prob = net.forward();
		//! [Make forward pass]

		//! [Get a class with a highest score]
		Point classIdPoint;
		double confidence;
		minMaxLoc(prob.reshape(1, 1), 0, &confidence, 0, &classIdPoint);
		int classId = classIdPoint.x;
		//! [Get a class with a highest score]

		// Put efficiency information.
		std::vector<double> layersTimes;
		double freq = getTickFrequency() / 1000;
		double t = net.getPerfProfile(layersTimes) / freq;
		std::string label = format("Inference time: %.2f ms", t);
		putText(frame, label, Point(0, 15), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0), 2);

		// Print predicted class.
		label = format("%s: %.4f", (classes.empty() ? format("Class #%d", classId).c_str() :
			classes[classId].c_str()),
			confidence);
		putText(frame, label, Point(0, 40), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0), 2);

		imshow(kWinName, frame);
	}
	return 0;
}
