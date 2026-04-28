#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

using namespace std;
using namespace cv;

int main(int argc, char** argv) {
    if (argc < 3) {
        cout << "Uso: ./panorama_360 img1.jpg img2.jpg img3.jpg ..." << endl;
        return -1;
    }

    // Cargamos la primera imagen como base
    Mat panorama = imread(argv[1]);
    if (panorama.empty()) return -1;

    // Redimensionar base para la RPi 3 (ajusta según necesites)
    if (panorama.cols > 800) 
        resize(panorama, panorama, Size(), 0.6, 0.6);

    Ptr<ORB> orb = ORB::create(1500);
    BFMatcher matcher(NORM_HAMMING);

    for (int i = 2; i < argc; i++) {
        cout << "Procesando imagen " << i << " de " << argc - 1 << "..." << endl;
        Mat nextImg = imread(argv[i]);
        if (nextImg.empty()) continue;
        
        if (nextImg.cols > 800)
            resize(nextImg, nextImg, Size(), 0.6, 0.6);

        // 1. Detectar puntos
        vector<KeyPoint> kp1, kp2;
        Mat desc1, desc2;
        orb->detectAndCompute(panorama, noArray(), kp1, desc1);
        orb->detectAndCompute(nextImg, noArray(), kp2, desc2);

        // 2. Matching
        vector<DMatch> matches;
        matcher.match(desc1, desc2, matches);
        sort(matches.begin(), matches.end());
        matches.erase(matches.begin() + (matches.size() * 0.2), matches.end());

        // 3. Homografía
        vector<Point2f> pts1, pts2;
        for (auto& m : matches) {
            pts1.push_back(kp1[m.queryIdx].pt);
            pts2.push_back(kp2[m.trainIdx].pt);
        }
        Mat H = findHomography(pts2, pts1, RANSAC);

        // 4. Warping y Combinación
        Mat temp;
        // Expandimos el lienzo para la nueva imagen
        warpPerspective(nextImg, temp, H, Size(panorama.cols + nextImg.cols/2, panorama.rows));
        
        // Creamos un lienzo nuevo y pegamos el panorama anterior encima
        Mat final_step = temp.clone();
        Mat roi(final_step, Rect(0, 0, panorama.cols, panorama.rows));
        panorama.copyTo(roi);
        
        panorama = final_step.clone();
    }

    imwrite("panorama_360_result.jpg", panorama);
    cout << "¡Hecho! Resultado guardado." << endl;

    return 0;
}