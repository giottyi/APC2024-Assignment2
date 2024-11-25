#include "convolutional_layer.hpp"

namespace convnet {

    convolutional_layer::convolutional_layer(std::size_t _s_filter, std::size_t _prev_depth, std::size_t _n_filters,
                                             std::size_t _s_stride, std::size_t _s_padding)
            : s_filter(_s_filter), prev_depth(_prev_depth), n_filters(_n_filters), s_stride(_s_stride),
              s_padding(_s_padding) {
        initialize();
    }

    void convolutional_layer::initialize() {
        for (std::size_t it = 0; it < n_filters; ++it) {
            tensor_3d filter(s_filter, s_filter, prev_depth);
            filter.initialize_with_random_normal(0.0, 3.0 / (2 * s_filter + prev_depth));
            filters.push_back(filter);
        }
    }

    tensor_3d convolutional_layer::evaluate(const tensor_3d &inputs) const {

        /* YOUR CODE SHOULD GO HERE */

		const std::size_t height_out = (inputs.get_height() - s_filter + 2 * s_padding) / s_stride + 1;
        const std::size_t width_out = (inputs.get_width() - s_filter + 2 * s_padding) / s_stride + 1;
        tensor_3d output(height_out, width_out, n_filters);
        output.initialize_with_zeros();

        const std::size_t filter_depth = inputs.get_depth(); // depths must match for a 2d filter
        for(std::size_t i = 0; i < height_out; ++i) {
            for(std::size_t j = 0; j < width_out; ++j) {
                for(std::size_t k = 0; k < n_filters; ++k) {
                    double result = 0.0;
                    for(std::size_t h = 0; h < s_filter; ++h) {
                        for(std::size_t w = 0; w < s_filter; ++w) {
                            for(std::size_t d = 0; d < filter_depth; ++d) {
                                result += inputs(i*s_stride+h,j*s_stride+w,d) * filters[k](h,w,d);
                            }
                        }
                    } output(i,j,k) = result;
                }
            }
        } return output;

    }

    tensor_3d convolutional_layer::apply_activation(const tensor_3d &Z) const {
        return act_function.apply(Z);
    }

    tensor_3d convolutional_layer::forward_pass(const tensor_3d &inputs) const {

        /* YOUR CODE SHOULD GO HERE */

		return act_function.apply(evaluate(inputs));

    }


    std::vector<std::vector<double>> convolutional_layer::get_parameters() const {
        std::vector<std::vector<double> > parameters;
        for (tensor_3d filter: filters) {
            parameters.push_back(filter.get_values());
        }
        return parameters;
    }

    void convolutional_layer::set_parameters(const std::vector<std::vector<double>> parameters) {
        for (std::size_t i = 0; i < n_filters; ++i) {
            filters[i].set_values(parameters[i]);
        }
    }

} // namespace
