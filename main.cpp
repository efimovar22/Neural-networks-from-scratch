#include <NetWork/NetWork.h>
#include <iostream>
#include <chrono>

struct data_info {
    double* pixels;
    int digit;
};

data_NetWork ReadDataNetWork(std::string path) {
    data_NetWork data{};
    std::ifstream fin;
    fin.open(path);
    if (!fin.is_open()) {
        std::cout << "Error reading the file " << path << std::endl;
        system("pause");
    }
    else
        std::cout << path << " loading...\n";
    std::string tmp;
    int L;
    while (!fin.eof()) {
        fin >> tmp;
        if (tmp == "NetWork") {
            fin >> L;
            data.L = L;
            data.size = new int[L];
            for (int i = 0; i < L; i++) {
                fin >> data.size[i];
            }
        }
    }
    fin.close();
    return data;
}

data_info* ReadData(std::string path, const data_NetWork& data_NW, int &examples) {
    data_info* data;
    std::ifstream fin;
    fin.open(path);
    if (!fin.is_open()) {
        std::cout << "Error reading the file " << path << std::endl;
        system("pause");
    }
    else
        std::cout << path << " loading... \n";
    std::string tmp;
    fin >> tmp;
    if (tmp == "Examples") {
        fin >> examples;
        std::cout << "Examples: " << examples << std::endl;
        data = new data_info[examples];
        for (int i = 0; i < examples; ++i)
            data[i].pixels = new double[data_NW.size[0]];

        for (int i = 0; i < examples; ++i) {
            fin >> data[i].digit;
            for (int j = 0; j < data_NW.size[0]; ++j) {
                fin >> data[i].pixels[j];
            }
        }
        fin.close();
        std::cout << "lib_MNIST loaded... \n";
        return data;
    }
    else {
        std::cout << "Error loading: " << path << std::endl;
        fin.close();
        return nullptr;
    }
}

int main() {
    NetWork NW{};
    data_NetWork NW_config;
    data_info* data;
    double ra = 0, right, predict, maxra = 0;
    int epoch = 0;
    bool study, repeat = true;
    std::chrono::duration<double> time;

    NW_config = ReadDataNetWork("Config.txt");
    NW.Init(NW_config);
    NW.PrintConfig();

    while (repeat) {
        std::cout << "STUDY? (1/0)" << std::endl;
        std::cin >> study;
        if (study) {
            int examples;
            data = ReadData("lib_MNIST_edit.txt", NW_config, examples);
            auto begin = std::chrono::steady_clock::now();
            while (ra / examples * 100 < 100) {
                ra = 0;
                auto t1 = std::chrono::steady_clock::now();
                for (int i = 0; i < examples; ++i) {
                    NW.SetInput(data[i].pixels);
                    right = data[i].digit;
                    predict = NW.ForwardFeed();
                    if (predict != right) {
                        NW.BackPropogation(right);
                        NW.WeightsUpdater(0.15 * exp(-epoch / 20.));
                    }
                    else 
                        ra++;
                }
                auto t2 = std::chrono::steady_clock::now();
                time = t2 - t1;
                if (ra > maxra) maxra = ra;
                std::cout << "ra: " << ra / examples * 100 << "\t" << "maxra: " << maxra / examples * 100 << "\t" << "epoch: " << epoch << "\tTIME: " << time.count() << std::endl;
                epoch++;
                if (epoch == 20) 
                    break;
            }
            auto end = std::chrono::steady_clock::now();
            time = end - begin;
            std::cout << "TIME: " << time.count() / 60. << " min" << std::endl;
            NW.SaveWeights();
        }
        else {
            NW.ReadWeights();
        }
        std::cout << "Test? (1/0)\n";
        bool test_flag;
        std::cin >> test_flag;
        if (test_flag) {
            int ex_tests;
            data_info* data_test;
            data_test = ReadData("lib_10k.txt", NW_config, ex_tests);
            ra = 0;
            for (int i = 0; i < ex_tests; ++i) {
                NW.SetInput(data_test[i].pixels);
                predict = NW.ForwardFeed();
                right = data_test[i].digit;
                if (right == predict) 
                    ra++;
            }
            std::cout << "RA: " << ra / ex_tests * 100 << std::endl;
        }
        std::cout << "Repeat? (1/0)\n";
        std::cin >> repeat;
    }
    system("pause");
    return 0;
}
