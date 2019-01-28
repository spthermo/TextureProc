#pragma once

#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>


//Sensor parameters parsing - KRT files
bool read_ext_file(const char* filename, std::vector< std::vector<float>>& R, std::vector<float>& T, std::vector< std::vector<float>>& cam_params)
{
	auto is_valid = false;

	if (filename != "")
	{
		R = std::vector< std::vector<float>>(3, std::vector<float>(3, 0));
		T = std::vector<float>(3, 0);
		cam_params = std::vector< std::vector<float>>(2, std::vector<float>(4, 0));

		std::ifstream param_file = std::ifstream(filename);

		std::string line, token;

		for (int i = 0; i < 5; std::getline(param_file, line), ++i) // 5 is the number of lines + 1
		{
			is_valid = false;

			if (line == "")
				continue;

			std::istringstream iss(line);

			std::vector<double> line_values;

			while (std::getline(iss, token, ' ')) {
				line_values.push_back(std::atof(token.c_str()));
				// std::cout << token << '\n';
			}

			switch (i)
			{
			case 1:
				cam_params[0][0] = line_values[0]; cam_params[0][1] = line_values[1];
				cam_params[0][2] = line_values[2]; cam_params[0][3] = line_values[3];
				break;
			case 2:
				cam_params[1][0] = line_values[0]; cam_params[1][1] = line_values[4];
				cam_params[1][2] = line_values[2]; cam_params[1][3] = line_values[5];
				break;
			case 3:
				R[0][0] = line_values[0]; R[0][1] = line_values[1]; R[0][2] = line_values[2];
				R[1][0] = line_values[3]; R[1][1] = line_values[4]; R[1][2] = line_values[5];
				R[2][0] = line_values[6]; R[2][1] = line_values[7]; R[2][2] = line_values[8];
				break;
			case 4:
				T[0] = line_values[0]; T[1] = line_values[1]; T[2] = line_values[2];
				break;
			default:
				break;
			}
			// std::cout << iss.str();
			is_valid = true;
			// std::getchar();
		}
	}


	return is_valid;
}

template<typename T> // Can be any type of number
double CalcMedian(std::vector<T> scores)
{
	size_t size = scores.size();

	if (size == 0)
	{
		return 0;  // Undefined, really.
	}
	else
	{
		sort(scores.begin(), scores.end());
		if (size % 2 == 0)
		{
			return (scores[size / 2 - 1] + scores[size / 2]) / 2;
		}
		else
		{
			return scores[size / 2];
		}
	}
}