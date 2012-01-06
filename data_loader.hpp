#ifndef _CPP_DATA_LOADER_
#define _CPP_DATA_LOADER_

#include <iostream>
#include <fstream>
#include <string>
#include <sstream>

#include <stdio.h>
#include <stdlib.h>

#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/io.hpp>

#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_int.hpp>

// refer to matrix row
#include <boost/numeric/ublas/matrix_proxy.hpp>


#include "util.hpp"

class SimpleDataLoader {
    private:
        int record_num;
        int dim_num;

        bool debug;

    private:


        int get_cat(const string& data) {
            int c;
            convert_from_string(c, data);

            return c;
        }

        bool get_features(const string& data, int& index, double& value) {
            int pos = data.find(":");
            if (pos == -1) return false;
            convert_from_string(index, data.substr(0, pos));
            convert_from_string(value, data.substr(pos + 1));

            return true;
        }

        // please note we need to add a default feature to each instance and set the feature weight to 1
        bool parse_line(const string& line, int& cat, const int line_num, boost::numeric::ublas::matrix<double>& x) {
            if (line.empty()) return false;
            size_t start_pos = 0;
            char space = ' ';
            
            // the dummy feature 
            x(line_num, 0) = 1;

            while (true) {
                size_t pos = line.find(space, start_pos);
                string data = line.substr(start_pos, pos - start_pos);
                if (!data.empty()) {
                    if (start_pos == 0) {
                        cat = get_cat(data);
                    }
                    else {
                        int index = -1;
                        double v = 0;
                        get_features(data, index, v);
                        if (debug)
                            cout << "index: " << index << "," << "value: " << v << endl;
                        if (index != -1) {
                            x(line_num, index) = v;
                        }
                    }
                }
                if ((int)pos != -1) {
                    start_pos = pos + 1;
                }
                else {
                    break;
                }
            }

            return true;

        }


    public:
        SimpleDataLoader(const int r, const int c) : record_num(r), dim_num(c), debug(false) {}

        void load_file(char*& file_path, boost::numeric::ublas::vector<double>& y, boost::numeric::ublas::matrix<double>& x) {
            ifstream in(file_path);
            string line;
            int line_num = 0;
            if (in.is_open()) {
                while (in.good()) {
                    getline(in, line);
                    if (line.empty()) continue;
                    int cat = 0;
                    if (!parse_line(line, cat, line_num, x)) {
                        cout << "parse line: " << line << ", failed.." << endl;
                        continue;
                    }

                    y(line_num) = cat;

                    line_num += 1;
                }
                in.close();
            }

        }
};


#endif
