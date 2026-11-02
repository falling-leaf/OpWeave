CREATE OR REPLACE FUNCTION load_base_model(OUT void)
    AS 'MODULE_PATHNAME', 'load_base_model' LANGUAGE C STRICT;

CREATE TABLE IF NOT EXISTS ai_operator(operator text primary key, discription text);
insert into ai_operator values('Add','element-wise addition of two tensors');
insert into ai_operator values('Subtract','element-wise subtraction of two tensors');
insert into ai_operator values('Multiply','element-wise multiplication');
insert into ai_operator values('Divide','element-wise division');
insert into ai_operator values('Pow','element-wise exponentiation');
insert into ai_operator values('Exp','element-wise exponential');
insert into ai_operator values('Abs','element-wise absolute value');
insert into ai_operator values('Sqrt','element-wise square root');
insert into ai_operator values('ReLU','f(x) = max(0, x)');
insert into ai_operator values('Sigmoid','f(x) = 1 / (1 + exp(-x))');
insert into ai_operator values('Softmax','normalizes the input into a probability distribution, commonly used in the output layer for classification tasks.');
insert into ai_operator values('Softplus','f(x) = log(1 + exp(x))');
insert into ai_operator values('MatMul','matrix multiplication');
insert into ai_operator values('Linear','y = Wx + b');
insert into ai_operator values('Convolution','slides the convolution kernel over the input data to extract features');
insert into ai_operator values('Max Pooling','takes the maximum value in a local region');
insert into ai_operator values('Average Pooling','takes the average value in a local region');
insert into ai_operator values('Layer Normalization','normalizes the input of each layer');
insert into ai_operator values('MSE','L = (y_pred - y_true)^2');
insert into ai_operator values('Gradient Calculation','computes the gradient of the loss function with respect to the parameters');
insert into ai_operator values('L1 Loss','L = |y_pred - y_true|');
insert into ai_operator values('L2 Loss','L = (y_pred - y_true)^2');

CREATE OR REPLACE FUNCTION api_load_model(  
    model_name cstring,
    model_path cstring
)  
RETURNS boolean
AS 'MODULE_PATHNAME', 'api_load_model'  
LANGUAGE C STRICT;

CREATE OR REPLACE FUNCTION api_predict(  
    model_name cstring,
    image_url cstring
)  
RETURNS boolean
AS 'MODULE_PATHNAME', 'api_predict'  
LANGUAGE C STRICT;


CREATE OR REPLACE FUNCTION db_agent_final(internal)
RETURNS float8
AS 'MODULE_PATHNAME', 'db_agent_final'
LANGUAGE C;

CREATE OR REPLACE FUNCTION db_agent_sfinal(internal)
RETURNS float8[]
AS 'MODULE_PATHNAME', 'db_agent_sfinal'
LANGUAGE C;

CREATE OR REPLACE FUNCTION db_agent_sfunc(internal, cstring, VARIADIC "any")
RETURNS internal
AS 'MODULE_PATHNAME', 'db_agent_sfunc' -- corresponds to the new entry name in the C code
LANGUAGE C;

CREATE OR REPLACE FUNCTION db_agent_msfunc(IN aggstate internal, IN task_type cstring, VARIADIC vec "any" , OUT internal)
    AS 'MODULE_PATHNAME', 'db_agent_msfunc' LANGUAGE C;
CREATE OR REPLACE FUNCTION db_agent_minvfunc(IN aggstate internal, IN task_type cstring, VARIADIC vec "any" , OUT internal)
    AS 'MODULE_PATHNAME', 'db_agent_minvfunc' LANGUAGE C;
CREATE OR REPLACE FUNCTION db_agent_mfinalfunc(IN aggstate internal, OUT float8)
    AS 'MODULE_PATHNAME', 'db_agent_mfinalfunc' LANGUAGE C;

CREATE AGGREGATE db_agent_batch(cstring, VARIADIC "any") (
    STYPE = internal,
    SFUNC = db_agent_sfunc,
    FINALFUNC = db_agent_final,
    MSTYPE=internal,
    MSFUNC = db_agent_msfunc,
    MINVFUNC = db_agent_minvfunc,
    MFINALFUNC = db_agent_mfinalfunc
);

CREATE AGGREGATE db_agent_single(cstring, VARIADIC "any") (
    STYPE = internal,
    SFUNC = db_agent_sfunc,
    FINALFUNC = db_agent_sfinal
);

-- insert into base_model_info values('resnet18', 'd17e88193d63653e785ccd1e8314caee', '/home/lhh/models/resnet18_resnet18_imagenet.pt');
-- insert into base_model_info values('alexnet', '722de753346c94d79e7397a28c6f5674', '/home/lhh/models/alexnet_alexnet_imagenet.pt');
-- insert into base_model_info values('googlenet', '6206b310ee0c21ecdb07516645b5c686', '/home/lhh/models/googlenet_googlenet_imagenet.pt');
-- insert into base_model_info values('resnet50', '2e654b8cc31f235b0bc60bd1852517c4', '/home/lhh/models/resnet50_resnet50_voc2007.pt');