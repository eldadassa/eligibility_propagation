//#include <ale/ale_interface.hpp>
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/lib/random/simple_philox.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/util/guarded_philox_random.h"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

using namespace tensorflow;

REGISTER_OP("Optest")
    .Attr("rom_file: string")
    .Attr("frameskip_min: int = 2")
    .Attr("frameskip_max: int = 5")
    .Attr("seed: int = 0")
    .Attr("seed2: int = 0")
    .Input("action: int32")
    .Input("reset: int32")
    .Input("max_episode_len: int32")
    .Output("reward: float")
    .Output("done: bool")
    .Output("screen: uint8")
    .SetIsStateful()
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->Scalar());
      c->set_output(1, c->Scalar());
      // no shape inference for screen, because we don't know screen dimensions yet
      return Status::OK();
    })
    .Doc(R"doc(OP test)doc");


class TestOp : public OpKernel {
 public:
  explicit TestOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context,
                   context->GetAttr("rom_file", &rom_file_));
    OP_REQUIRES_OK(context,
		   Env::Default()->FileExists(rom_file_));

    //ale_.loadROM(rom_file_);
    OP_REQUIRES_OK(context,
                   context->GetAttr("frameskip_min", &frameskip_min_));
    OP_REQUIRES_OK(context,
                   context->GetAttr("frameskip_max", &frameskip_max_));
    OP_REQUIRES(context, frameskip_min_ > 0,
                errors::InvalidArgument("frameskip_min must be > 0"));
    OP_REQUIRES(context, frameskip_max_ >= frameskip_min_,
                errors::InvalidArgument("frameskip_max must be >= frameskip_min"));

    //auto legalActionsVec = ale_.getLegalActionSet();
    std::vector<int32> legalActionsVec{0, 1, 2};
    counter_ = 0;
    std::copy(legalActionsVec.begin(), legalActionsVec.end(), std::inserter(legalActions_, legalActions_.end()));
    OP_REQUIRES_OK(context, generator_.Init(context));

    cam_heading_ = 0;
    counter_no_reward_ = 0;
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& input_tensor = context->input(0);
    OP_REQUIRES(context, TensorShapeUtils::IsScalar(input_tensor.shape()),
                errors::InvalidArgument("Ale expects scalar action."));
    auto input = input_tensor.scalar<int32>();

    //ale::Action action = (ale::Action) input(0);
    int32 action = (int32) input(0);

    OP_REQUIRES(context, legalActions_.find(action) != legalActions_.end(),
                errors::InvalidArgument("Action is out of legal actions range."));

    const Tensor& max_episode_length_tensor = context->input(2);
    OP_REQUIRES(context, TensorShapeUtils::IsScalar(max_episode_length_tensor.shape()),
                errors::InvalidArgument("Ale expects scalar maximum episode length."));
    auto max_episode_length = max_episode_length_tensor.scalar<int32>()(0);

    const Tensor& reset_tensor = context->input(1);
    OP_REQUIRES(context, TensorShapeUtils::IsScalar(reset_tensor.shape()),
                errors::InvalidArgument("Ale expects scalar reset."));
    auto should_reset = reset_tensor.scalar<int32>()(0);

    //const int w = ale_.getScreen().width();
    //const int h = ale_.getScreen().height();
    const int w = 160;
    const int h = 210;

    auto local_gen = generator_.ReserveSamples32(1);
    random::SimplePhilox random(&local_gen);
    int to_repeat = frameskip_min_ + random.Uniform(frameskip_max_ - frameskip_min_);

    if(should_reset) {
      //ale_.reset_game();
      counter_ = 0;
      //int no_ops = random.Uniform(30);
      //for (int i = 0; i < no_ops; i++) {
      //    ale_.act((ale::Action) 0);
      //    }
      cam_heading_ = random.Uniform(120)-60;
      counter_no_reward_ = 0;
    }

    //float r = 0.0;
    for(;to_repeat > 0; --to_repeat){
      //r += ale_.act(action);
      act(action);
    }
    counter_ += 1;

    //bool done = ale_.game_over() || (max_episode_length > 0 && counter_ >= max_episode_length);
    bool done = counter_no_reward_>200 || (max_episode_length > 0 && counter_ >= max_episode_length);
    if(done) {
      //ale_.reset_game();
      counter_ = 0;
      //int no_ops = random.Uniform(30);
      //for (int i = 0; i < no_ops; i++) {
      //    ale_.act((ale::Action) 0);
      //}
      cam_heading_ = random.Uniform(120)-60;
      counter_no_reward_ = 0;
    }

    Tensor* reward_tensor = NULL;
    Tensor* done_tensor = NULL;
    Tensor* screen_tensor = NULL;

    OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape({}),
                                                     &reward_tensor));
    OP_REQUIRES_OK(context, context->allocate_output(1, TensorShape({}),
						     &done_tensor));
    OP_REQUIRES_OK(context, context->allocate_output(2, TensorShape({h, w, 3}),
						     &screen_tensor));

    auto output_r = reward_tensor->scalar<float>();
    auto output_d = done_tensor->scalar<bool>();
    //auto output_s = screen_tensor->flat<unsigned char>();
    auto output_tensor_mapped = screen_tensor->tensor<unsigned char, 3>();

    //std::vector<unsigned char> screen_buff(h*w*3, 0);
    //ale_.getScreenRGB(screen_buff);
    
    float r = 0.0;
    cv::Mat cvScreen = cv::Mat::zeros(h,w,CV_8UC3);
    r = getScreenRGB(cvScreen, action);
    if (r<=0.0)
       counter_no_reward_++;
    else
       counter_no_reward_=0;

    output_r(0) = r;
    output_d(0) = done;
    //ale_.getScreenRGB(output_s.data());
    //std::copy_n(screen_buff.begin(), h * w * 3,
    //		output_s.data()); // get rid of copy?

    for (int y = 0; y < cvScreen.rows; y++) {
       for (int x = 0; x < cvScreen.cols; x++) {
          cv::Vec3b pixel = cvScreen.at<cv::Vec3b>(y, x);

          output_tensor_mapped( y, x, 0) = pixel.val[2]; //R
          output_tensor_mapped( y, x, 1) = pixel.val[1]; //G
          output_tensor_mapped( y, x, 2) = pixel.val[0]; //B
       }
    }
  }

private:
  //ale::ALEInterface ale_;
  //std::set<ale::Action> legalActions_;
  std::set<int32> legalActions_;
  std::string rom_file_;
  int frameskip_min_;
  int frameskip_max_;
  int counter_;
  GuardedPhiloxRandom generator_;
  int cam_heading_;
  int counter_no_reward_;

  void act(int action){
     // moving cam heading according to current action
     if (action==0)
       cam_heading_--;
    else if (action==2)
       cam_heading_++;

    // enabling cyclic env
    if (cam_heading_>160)
       cam_heading_=-160;
    else if (cam_heading_<-160)
       cam_heading_=160;
  }

  float getScreenRGB(cv::Mat &image, int action){
    float reward = 0.0;
    if (action==1)
       return reward;

    //draw line
    int draw_line_at = 0;
    if (cam_heading_>=-20 && cam_heading_<20){
      draw_line_at = image.cols/2 - cam_heading_*(image.cols/40) - 1;

      int thickness = 2;
      int lineType = cv::LINE_8;
      cv::line( image, cv::Point(draw_line_at,0), cv::Point(draw_line_at,image.rows-1),
                cv::Scalar(250, 250, 250), thickness, lineType );

      reward = 1.0;
    }
    else {
      reward = -1.0;
    }
    
    return reward;
  }

};

REGISTER_KERNEL_BUILDER(Name("Optest").Device(DEVICE_CPU), TestOp);
