#include "max_pooling_layer.hpp"

namespace convnet {

    max_pooling_layer::max_pooling_layer(std::size_t s_filter, std::size_t strd) {
        size_filter = s_filter;
        stride = strd;
    };

    tensor_3d max_pooling_layer::evaluate(const tensor_3d &inputs) const {

        /* YOUR CODE SHOULD GO HERE */

		const std::size_t height_out = (inputs.get_height() - size_filter) / stride + 1;
        const std::size_t width_out = (inputs.get_width() - size_filter) / stride + 1;
        const std::size_t depth_out = inputs.get_depth(); // depths must match for a 2d filter
        tensor_3d output(height_out, width_out, depth_out);
		output.initialize_with_zeros();

        for(std::size_t i = 0; i < height_out; ++i) {
            for(std::size_t j = 0; j < width_out; ++j) {
                for(std::size_t k = 0; k < depth_out; ++k) {
                    double max = 0.0;
                    for(std::size_t h = 0; h < size_filter; ++h ) {
                        for(std::size_t w = 0; w < size_filter; ++w ) {
                            max = (max > inputs(i*stride+h,j*stride+w,k)) ? max : inputs(i*stride+h,j*stride+w,k);
                        }
                    } output(i,j,k) = max;
                }
            }
        } return output;

    };


    tensor_3d max_pooling_layer::apply_activation(const tensor_3d &Z) const {
        return Z;
    };

    tensor_3d max_pooling_layer::forward_pass(const tensor_3d &inputs) const {

        /* YOUR CODE SHOULD GO HERE */

		// can't return result directly, address of a local temporary object may escape the function error
        tensor_3d result = apply_activation(evaluate(inputs));
        return result;

    };

    // Do nothing since max pooling has no learnable parameter
    void max_pooling_layer::set_parameters(const std::vector<std::vector<double>> parameters) {}

} // namespace
