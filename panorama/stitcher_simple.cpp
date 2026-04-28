#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <iostream>
#include <vector>

using namespace std;
using namespace cv;

int main(int argc, char** argv) {
    if (argc < 3) {
        cout << "Uso: ./stitcher_simple imagen1.jpg imagen2.jpg" << endl;
        return -1;
    }

    // 1. Cargar imágenes
    Mat img1 = imread(argv[1]);
    Mat img2 = imread(argv[2]);

    // Redimensionar si son muy grandes para ahorrar RAM en la Pi 3
    if (img1.cols > 1000) {
        resize(img1, img1, Size(), 0.5, 0.5);
        resize(img2, img2, Size(), 0.5, 0.5);
    }

    // 2. Detectar puntos clave y descriptores con ORB (Optimizado para ARM)
    Ptr<ORB> orb = ORB::create(1000);
    vector<KeyPoint> keypoints1, keypoints2;
    Mat descriptors1, descriptors2;

    orb->detectAndCompute(img1, noArray(), keypoints1, descriptors1);
    orb->detectAndCompute(img2, noArray(), keypoints2, descriptors2);

    // 3. Emparejar puntos (Matching)
    BFMatcher matcher(NORM_HAMMING);
    vector<DMatch> matches;
    matcher.match(descriptors1, descriptors2, matches);

    // Filtrar los mejores matches
    sort(matches.begin(), matches.end());
    matches.erase(matches.begin() + (matches.size() * 0.3), matches.end());

    // 4. Calcular Homografía (RANSAC)
    vector<Point2f> points1, points2;
    for (auto& m : matches) {
        points1.push_back(keypoints1[m.queryIdx].pt);
        points2.push_back(keypoints2[m.trainIdx].pt);
    }

    Mat H = findHomography(points2, points1, RANSAC);

    // 5. Unir imágenes (Warping)
    Mat result;
    warpPerspective(img2, result, H, Size(img1.cols + img2.cols, img1.rows));
    
    Mat half(result, Rect(0, 0, img1.cols, img1.rows));
    img1.copyTo(half);

    // Guardar resultado
    imwrite("resultado_pi.jpg", result);
    cout << "Panorama guardado como resultado_pi.jpg" << endl;

    return 0;
}