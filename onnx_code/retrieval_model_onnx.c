#include <stdio.h>
#include <stdlib.h>
#include <onnxruntime_c_api.h>

// Global variables for the ONNX Runtime environment, session, and input/output tensors
OrtEnv* env;
OrtSession* session;
OrtSessionOptions* session_options;
//OrtAllocator* allocator;
OrtMemoryInfo* memory_info;
OrtStatus *status;
// Initialization function
OrtApi *g_api[4];


void init_onnx_runtime_(void) {
    // Initialize the ONNX Runtime environment
    const OrtApi *api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    //"conv_and_strat_model_kuka_jan24_2025.onnx"
    for(int im=0;im<4;im++)
      {
	char model_path[150];
	if(im==0)
	  strcpy(model_path,"onnx_density_net_models/conv_st_ku_densi_net_feb16_2025_01.onnx");
	if(im==1)
	  strcpy(model_path,"onnx_density_net_models/conv_ku_densi_net_feb16_2025_01.onnx");
	if(im==2)
	  strcpy(model_path,"onnx_density_net_models/conv_st_ku_densi_net_feb16_2025_01.onnx");
	if(im==3)
	  strcpy(model_path,"onnx_density_net_models/conv_ku_densi_net_feb16_2025_01.onnx");
	g_api[im] = (OrtApi *)api;
	status = g_api[im]->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "test", &env);
	status = g_api[im]->CreateSessionOptions(&session_options);
	status = g_api[im]->SetIntraOpNumThreads(session_options, 1);
	status = g_api[im]->CreateSession(env, model_path, session_options, &session);
	if (status != NULL) {
	  const char* msg = g_api[im]->GetErrorMessage(status);
	  fprintf(stderr, "Failed to run ONNX model: %s\n", msg);
	  g_api[im]->ReleaseStatus(status);
	}
	status = g_api[im]->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memory_info);
      }

}

void call_onnx_(float *input_data, int *lengths_data, float *output_data, int *batch_size, int *seq_len, int *input_size, int *output_size, int *im, float *input_data2, float *output_data2, int *output_2_size) {
    //'input': input_data, 'n_seq'
    const char* input_names[] = {"input_1", "n_seq", "input_2"};
    const char* output_names[] = {"output1", "output2"};

    int64_t input_1_shape[3] = {(int64_t)(*batch_size), (int64_t)(*seq_len), (int64_t)(*input_size)};
    int64_t input_2_shape[2] = {(int64_t)(*batch_size), (int64_t)(1)};
    size_t input_1_tensor_size = (*batch_size) * (*seq_len) * (*input_size);
    size_t input_2_tensor_size = (*batch_size);
    //printf("%i %i %i %i %i %i\n", *batch_size, *seq_len, *input_size, *output_size, *im, *output_2_size);
    OrtValue* input_1_tensor;
    OrtValue* input_2_tensor;
    OrtValue* output_1_tensor;
    OrtValue* output_2_tensor;
    
    //OrtValue* input_tensor = NULL;

    status=g_api[*im]->CreateTensorWithDataAsOrtValue(
        memory_info, input_data, input_1_tensor_size * sizeof(float),
        input_1_shape, 3, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &input_1_tensor);

    status=g_api[*im]->CreateTensorWithDataAsOrtValue(
        memory_info, input_data2, input_2_tensor_size * sizeof(float),
        input_2_shape, 2, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &input_2_tensor);

    if(status!=NULL)
      {
        fprintf(stderr, "Failed to create the input: %s\n", g_api[*im]->GetErrorMessage(status));
      }
    //printf("batch_size: %d\n", *batch_size);
    //printf("seq_len: %d\n", *seq_len);
    //printf("input_size: %d\n", *input_size);
    //printf("output_size: %d\n", *output_size);
    // Prepare seq_lengths tensor
    int64_t lengths_shape[1] = {(int32_t)(*batch_size)};
    OrtValue* lengths_tensor = NULL;
    int32_t *lengths_data_t;
    lengths_data_t = (int32_t *)malloc((*batch_size) * sizeof(int32_t));
    for (int i = 0; i < *batch_size; i++) {
        lengths_data_t[i] = (int32_t)lengths_data[i];
    }
    status=g_api[*im]->CreateTensorWithDataAsOrtValue(
        memory_info, lengths_data_t, (*batch_size) * sizeof(int32_t),
        lengths_shape, 1, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32, &lengths_tensor);
    
    if (status != NULL) {
        const char* msg = g_api[*im]->GetErrorMessage(status);
        fprintf(stderr, "Failed to run ONNX model: %s\n", msg);
        fprintf(stderr, "batch_size: %d\n", *batch_size);
        g_api[*im]->ReleaseStatus(status);
    }
    // Create input names and tensors
    const OrtValue* input_tensors[] = {input_1_tensor, lengths_tensor, input_2_tensor};

    // Prepare output tensor
    int64_t output_1_shape[3] = {(int64_t)(*batch_size), (int64_t)(*seq_len), (int64_t)(*output_size)};
    int64_t output_2_shape[2] = {(int64_t)(*batch_size), (int64_t)(*output_2_size)};
    size_t output_1_tensor_size = (*batch_size) * (*seq_len) * (*output_size);
    size_t output_2_tensor_size = (*batch_size) * (*output_2_size);
    output_1_tensor = NULL;
    output_2_tensor = NULL;
    
    status=g_api[*im]->CreateTensorWithDataAsOrtValue(
        memory_info, output_data, output_1_tensor_size * sizeof(float),
        output_1_shape, 3, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &output_1_tensor);

    if(status!=NULL)
      {
        fprintf(stderr, "Failed to create the output1: %s\n", g_api[*im]->GetErrorMessage(status));
      }
    status=g_api[*im]->CreateTensorWithDataAsOrtValue(
        memory_info, output_data2, output_2_tensor_size * sizeof(float),
        output_2_shape, 2, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &output_2_tensor);

    if(status!=NULL)
      {
        fprintf(stderr, "Failed to create the output2: %s\n", g_api[*im]->GetErrorMessage(status));
      }
    OrtValue* output_tensors[] = {output_1_tensor, output_2_tensor};
    // Run the model
    status = g_api[*im]->Run(
        session, NULL, input_names, input_tensors, 3,
        output_names, 2, output_tensors);

    if (status != NULL) {
        const char* msg = g_api[*im]->GetErrorMessage(status);
        fprintf(stderr, "Failed to run ONNX model: %s\n", msg);
        g_api[*im]->ReleaseStatus(status);
    }

    // Release resources
    g_api[*im]->ReleaseValue(input_1_tensor);
    g_api[*im]->ReleaseValue(input_2_tensor);
    g_api[*im]->ReleaseValue(lengths_tensor);
    g_api[*im]->ReleaseValue(output_1_tensor);
    g_api[*im]->ReleaseValue(output_2_tensor);
    free(lengths_data_t);
}
/*
int main(void)

{
    // Initialize the model
    
    init();

    // Prepare the input data
    float input_data[128 * 768];  // Adjust the input size as needed
    int32_t lengths_data[1] = {128};  // Adjust the sequence length as needed

    // Prepare the output data
    float output_data[128 * 768];  // Adjust the output size as needed

    // Run the prediction
    //predict(input_data, lengths_data, output_data, 1, 128, 768);

    // Print the output data
    //for (int i = 0; i < 128 * 768; i++) {
    //    printf("%f\n", output_data[i]);
    //}

    return 0;
}
*/
