#include <opencv2/opencv.hpp>
#include <opencv2/stitching.hpp> // El módulo oficial y estable
#include <iostream>
#include <vector>

using namespace std;
using namespace cv;

int main(int argc, char** argv) {
    if (argc < 3) {
        cout << "Uso: ./panorama_robusto img1.jpg img2.jpg img3.jpg ..." << endl;
        return -1;
    }

    vector<Mat> images;
    
    // 1. Cargar y optimizar imágenes
    for (int i = 1; i < argc; i++) {
        Mat img = imread(argv[i]);
        if (img.empty()) {
            cout << "No se pudo leer la imagen: " << argv[i] << endl;
            continue;
        }
        
        // REDIMENSIÓN VITAL PARA LA RASPBERRY PI 3
        // El Stitcher consume mucha RAM. Reducimos la imagen a un tamaño manejable.
        if (img.cols > 800) {
            resize(img, img, Size(), 0.5, 0.5); 
        }
        images.push_back(img);
    }

    cout << "Imágenes cargadas: " << images.size() << ". Iniciando fusión (esto puede tardar unos minutos en la RPi3)..." << endl;

    // 2. Crear el objeto Stitcher de OpenCV
    // Usamos el modo PANORAMA (ideal para giros de 360 desde un punto fijo)
    Ptr<Stitcher> stitcher = Stitcher::create(Stitcher::PANORAMA);
    
    // Ajuste opcional para webcam: si la calidad no es perfecta, bajamos la exigencia
    // stitcher->setPanoConfidenceThresh(0.8); 

    Mat panorama_final;
    
    // 3. Ejecutar el pipeline completo (Features, Matching, Bundle Adjustment, Warping, Blending)
    Stitcher::Status status = stitcher->stitch(images, panorama_final);

    if (status != Stitcher::OK) {
        cout << "Error al crear el panorama. Código de error: " << int(status) << endl;
        if (status == Stitcher::ERR_NEED_MORE_IMGS) {
            cout << "Causa probable: No hay suficiente solapamiento (overlap) entre las imágenes." << endl;
        }
        return -1;
    }

    // 4. Guardar resultado
    imwrite("panorama_perfecto.jpg", panorama_final);
    cout << "¡Éxito! Panorama guardado como panorama_perfecto.jpg" << endl;

    return 0;
}