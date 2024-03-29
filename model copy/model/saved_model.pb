��
��
D
AddV2
x"T
y"T
z"T"
Ttype:
2	��
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( �
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
�
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

�
FusedBatchNormV3
x"T

scale"U
offset"U	
mean"U
variance"U
y"T

batch_mean"U
batch_variance"U
reserve_space_1"U
reserve_space_2"U
reserve_space_3"U"
Ttype:
2"
Utype:
2"
epsilonfloat%��8"&
exponential_avg_factorfloat%  �?";
data_formatstringNHWC:
NHWCNCHWNDHWCNCDHW"
is_trainingbool(
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
�
Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(�

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype�
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
�
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ��
@
StaticRegexFullMatch	
input

output
"
patternstring
�
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
<
Sub
x"T
y"T
z"T"
Ttype:
2	
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.8.02v2.8.0-rc1-32-g3f878cff5b68�
|
dqn/dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*!
shared_namedqn/dense/kernel
u
$dqn/dense/kernel/Read/ReadVariableOpReadVariableOpdqn/dense/kernel*
_output_shapes

:@*
dtype0
t
dqn/dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedqn/dense/bias
m
"dqn/dense/bias/Read/ReadVariableOpReadVariableOpdqn/dense/bias*
_output_shapes
:*
dtype0
~
dqn/dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: 	*#
shared_namedqn/dense_1/kernel
w
&dqn/dense_1/kernel/Read/ReadVariableOpReadVariableOpdqn/dense_1/kernel*
_output_shapes

: 	*
dtype0
x
dqn/dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*!
shared_namedqn/dense_1/bias
q
$dqn/dense_1/bias/Read/ReadVariableOpReadVariableOpdqn/dense_1/bias*
_output_shapes
:	*
dtype0
~
conv2d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d/kernel
w
!conv2d/kernel/Read/ReadVariableOpReadVariableOpconv2d/kernel*&
_output_shapes
:*
dtype0
n
conv2d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d/bias
g
conv2d/bias/Read/ReadVariableOpReadVariableOpconv2d/bias*
_output_shapes
:*
dtype0
�
batch_normalization/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_namebatch_normalization/gamma
�
-batch_normalization/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization/gamma*
_output_shapes
:*
dtype0
�
batch_normalization/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_namebatch_normalization/beta
�
,batch_normalization/beta/Read/ReadVariableOpReadVariableOpbatch_normalization/beta*
_output_shapes
:*
dtype0
�
batch_normalization/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!batch_normalization/moving_mean
�
3batch_normalization/moving_mean/Read/ReadVariableOpReadVariableOpbatch_normalization/moving_mean*
_output_shapes
:*
dtype0
�
#batch_normalization/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#batch_normalization/moving_variance
�
7batch_normalization/moving_variance/Read/ReadVariableOpReadVariableOp#batch_normalization/moving_variance*
_output_shapes
:*
dtype0
�
conv2d_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv2d_1/kernel
{
#conv2d_1/kernel/Read/ReadVariableOpReadVariableOpconv2d_1/kernel*&
_output_shapes
:*
dtype0
r
conv2d_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_1/bias
k
!conv2d_1/bias/Read/ReadVariableOpReadVariableOpconv2d_1/bias*
_output_shapes
:*
dtype0
�
batch_normalization_1/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_1/gamma
�
/batch_normalization_1/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_1/gamma*
_output_shapes
:*
dtype0
�
batch_normalization_1/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_namebatch_normalization_1/beta
�
.batch_normalization_1/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_1/beta*
_output_shapes
:*
dtype0
�
!batch_normalization_1/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!batch_normalization_1/moving_mean
�
5batch_normalization_1/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_1/moving_mean*
_output_shapes
:*
dtype0
�
%batch_normalization_1/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*6
shared_name'%batch_normalization_1/moving_variance
�
9batch_normalization_1/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_1/moving_variance*
_output_shapes
:*
dtype0

NoOpNoOp
�B
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�A
value�AB�A B�A
�
	conv1
	conv2
flatten
state_value
	advantage
	variables
trainable_variables
regularization_losses
		keras_api

__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures*
�
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer-2
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses*
�
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer-2
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses*
�
 	variables
!trainable_variables
"regularization_losses
#	keras_api
$__call__
*%&call_and_return_all_conditional_losses* 
�

&kernel
'bias
(	variables
)trainable_variables
*regularization_losses
+	keras_api
,__call__
*-&call_and_return_all_conditional_losses*
�

.kernel
/bias
0	variables
1trainable_variables
2regularization_losses
3	keras_api
4__call__
*5&call_and_return_all_conditional_losses*
z
60
71
82
93
:4
;5
<6
=7
>8
?9
@10
A11
&12
'13
.14
/15*
Z
60
71
82
93
<4
=5
>6
?7
&8
'9
.10
/11*
* 
�
Bnon_trainable_variables

Clayers
Dmetrics
Elayer_regularization_losses
Flayer_metrics
	variables
trainable_variables
regularization_losses

__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
* 

Gserving_default* 
�

6kernel
7bias
H	variables
Itrainable_variables
Jregularization_losses
K	keras_api
L__call__
*M&call_and_return_all_conditional_losses*
�
Naxis
	8gamma
9beta
:moving_mean
;moving_variance
O	variables
Ptrainable_variables
Qregularization_losses
R	keras_api
S__call__
*T&call_and_return_all_conditional_losses*
�
U	variables
Vtrainable_variables
Wregularization_losses
X	keras_api
Y__call__
*Z&call_and_return_all_conditional_losses* 
.
60
71
82
93
:4
;5*
 
60
71
82
93*
* 
�
[non_trainable_variables

\layers
]metrics
^layer_regularization_losses
_layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
�

<kernel
=bias
`	variables
atrainable_variables
bregularization_losses
c	keras_api
d__call__
*e&call_and_return_all_conditional_losses*
�
faxis
	>gamma
?beta
@moving_mean
Amoving_variance
g	variables
htrainable_variables
iregularization_losses
j	keras_api
k__call__
*l&call_and_return_all_conditional_losses*
�
m	variables
ntrainable_variables
oregularization_losses
p	keras_api
q__call__
*r&call_and_return_all_conditional_losses* 
.
<0
=1
>2
?3
@4
A5*
 
<0
=1
>2
?3*
* 
�
snon_trainable_variables

tlayers
umetrics
vlayer_regularization_losses
wlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
�
xnon_trainable_variables

ylayers
zmetrics
{layer_regularization_losses
|layer_metrics
 	variables
!trainable_variables
"regularization_losses
$__call__
*%&call_and_return_all_conditional_losses
&%"call_and_return_conditional_losses* 
* 
* 
WQ
VARIABLE_VALUEdqn/dense/kernel-state_value/kernel/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEdqn/dense/bias+state_value/bias/.ATTRIBUTES/VARIABLE_VALUE*

&0
'1*

&0
'1*
* 
�
}non_trainable_variables

~layers
metrics
 �layer_regularization_losses
�layer_metrics
(	variables
)trainable_variables
*regularization_losses
,__call__
*-&call_and_return_all_conditional_losses
&-"call_and_return_conditional_losses*
* 
* 
WQ
VARIABLE_VALUEdqn/dense_1/kernel+advantage/kernel/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEdqn/dense_1/bias)advantage/bias/.ATTRIBUTES/VARIABLE_VALUE*

.0
/1*

.0
/1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
0	variables
1trainable_variables
2regularization_losses
4__call__
*5&call_and_return_all_conditional_losses
&5"call_and_return_conditional_losses*
* 
* 
MG
VARIABLE_VALUEconv2d/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE*
KE
VARIABLE_VALUEconv2d/bias&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUEbatch_normalization/gamma&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
XR
VARIABLE_VALUEbatch_normalization/beta&variables/3/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEbatch_normalization/moving_mean&variables/4/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUE#batch_normalization/moving_variance&variables/5/.ATTRIBUTES/VARIABLE_VALUE*
OI
VARIABLE_VALUEconv2d_1/kernel&variables/6/.ATTRIBUTES/VARIABLE_VALUE*
MG
VARIABLE_VALUEconv2d_1/bias&variables/7/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEbatch_normalization_1/gamma&variables/8/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEbatch_normalization_1/beta&variables/9/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUE!batch_normalization_1/moving_mean'variables/10/.ATTRIBUTES/VARIABLE_VALUE*
f`
VARIABLE_VALUE%batch_normalization_1/moving_variance'variables/11/.ATTRIBUTES/VARIABLE_VALUE*
 
:0
;1
@2
A3*
'
0
1
2
3
4*
* 
* 
* 
* 

60
71*

60
71*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
H	variables
Itrainable_variables
Jregularization_losses
L__call__
*M&call_and_return_all_conditional_losses
&M"call_and_return_conditional_losses*
* 
* 
* 
 
80
91
:2
;3*

80
91*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
O	variables
Ptrainable_variables
Qregularization_losses
S__call__
*T&call_and_return_all_conditional_losses
&T"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
U	variables
Vtrainable_variables
Wregularization_losses
Y__call__
*Z&call_and_return_all_conditional_losses
&Z"call_and_return_conditional_losses* 
* 
* 

:0
;1*

0
1
2*
* 
* 
* 

<0
=1*

<0
=1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
`	variables
atrainable_variables
bregularization_losses
d__call__
*e&call_and_return_all_conditional_losses
&e"call_and_return_conditional_losses*
* 
* 
* 
 
>0
?1
@2
A3*

>0
?1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
g	variables
htrainable_variables
iregularization_losses
k__call__
*l&call_and_return_all_conditional_losses
&l"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
m	variables
ntrainable_variables
oregularization_losses
q__call__
*r&call_and_return_all_conditional_losses
&r"call_and_return_conditional_losses* 
* 
* 

@0
A1*

0
1
2*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

:0
;1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

@0
A1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
�
serving_default_input_1Placeholder*/
_output_shapes
:���������*
dtype0*$
shape:���������
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1conv2d/kernelconv2d/biasbatch_normalization/gammabatch_normalization/betabatch_normalization/moving_mean#batch_normalization/moving_varianceconv2d_1/kernelconv2d_1/biasbatch_normalization_1/gammabatch_normalization_1/beta!batch_normalization_1/moving_mean%batch_normalization_1/moving_variancedqn/dense/kerneldqn/dense/biasdqn/dense_1/kerneldqn/dense_1/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������	*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *1
f,R*
(__inference_signature_wrapper_4374094562
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename$dqn/dense/kernel/Read/ReadVariableOp"dqn/dense/bias/Read/ReadVariableOp&dqn/dense_1/kernel/Read/ReadVariableOp$dqn/dense_1/bias/Read/ReadVariableOp!conv2d/kernel/Read/ReadVariableOpconv2d/bias/Read/ReadVariableOp-batch_normalization/gamma/Read/ReadVariableOp,batch_normalization/beta/Read/ReadVariableOp3batch_normalization/moving_mean/Read/ReadVariableOp7batch_normalization/moving_variance/Read/ReadVariableOp#conv2d_1/kernel/Read/ReadVariableOp!conv2d_1/bias/Read/ReadVariableOp/batch_normalization_1/gamma/Read/ReadVariableOp.batch_normalization_1/beta/Read/ReadVariableOp5batch_normalization_1/moving_mean/Read/ReadVariableOp9batch_normalization_1/moving_variance/Read/ReadVariableOpConst*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *,
f'R%
#__inference__traced_save_4374095032
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedqn/dense/kerneldqn/dense/biasdqn/dense_1/kerneldqn/dense_1/biasconv2d/kernelconv2d/biasbatch_normalization/gammabatch_normalization/betabatch_normalization/moving_mean#batch_normalization/moving_varianceconv2d_1/kernelconv2d_1/biasbatch_normalization_1/gammabatch_normalization_1/beta!batch_normalization_1/moving_mean%batch_normalization_1/moving_variance*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� */
f*R(
&__inference__traced_restore_4374095090��

�	
�
:__inference_batch_normalization_1_layer_call_fn_4374094902

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *^
fYRW
U__inference_batch_normalization_1_layer_call_and_return_conditional_losses_4374093655�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+���������������������������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
�	
�
:__inference_batch_normalization_1_layer_call_fn_4374094915

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *^
fYRW
U__inference_batch_normalization_1_layer_call_and_return_conditional_losses_4374093686�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+���������������������������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
�)
�
#__inference__traced_save_4374095032
file_prefix/
+savev2_dqn_dense_kernel_read_readvariableop-
)savev2_dqn_dense_bias_read_readvariableop1
-savev2_dqn_dense_1_kernel_read_readvariableop/
+savev2_dqn_dense_1_bias_read_readvariableop,
(savev2_conv2d_kernel_read_readvariableop*
&savev2_conv2d_bias_read_readvariableop8
4savev2_batch_normalization_gamma_read_readvariableop7
3savev2_batch_normalization_beta_read_readvariableop>
:savev2_batch_normalization_moving_mean_read_readvariableopB
>savev2_batch_normalization_moving_variance_read_readvariableop.
*savev2_conv2d_1_kernel_read_readvariableop,
(savev2_conv2d_1_bias_read_readvariableop:
6savev2_batch_normalization_1_gamma_read_readvariableop9
5savev2_batch_normalization_1_beta_read_readvariableop@
<savev2_batch_normalization_1_moving_mean_read_readvariableopD
@savev2_batch_normalization_1_moving_variance_read_readvariableop
savev2_const

identity_1��MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part�
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : �
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: �
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B-state_value/kernel/.ATTRIBUTES/VARIABLE_VALUEB+state_value/bias/.ATTRIBUTES/VARIABLE_VALUEB+advantage/kernel/.ATTRIBUTES/VARIABLE_VALUEB)advantage/bias/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*5
value,B*B B B B B B B B B B B B B B B B B �
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0+savev2_dqn_dense_kernel_read_readvariableop)savev2_dqn_dense_bias_read_readvariableop-savev2_dqn_dense_1_kernel_read_readvariableop+savev2_dqn_dense_1_bias_read_readvariableop(savev2_conv2d_kernel_read_readvariableop&savev2_conv2d_bias_read_readvariableop4savev2_batch_normalization_gamma_read_readvariableop3savev2_batch_normalization_beta_read_readvariableop:savev2_batch_normalization_moving_mean_read_readvariableop>savev2_batch_normalization_moving_variance_read_readvariableop*savev2_conv2d_1_kernel_read_readvariableop(savev2_conv2d_1_bias_read_readvariableop6savev2_batch_normalization_1_gamma_read_readvariableop5savev2_batch_normalization_1_beta_read_readvariableop<savev2_batch_normalization_1_moving_mean_read_readvariableop@savev2_batch_normalization_1_moving_variance_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
2�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*�
_input_shapes�
�: :@:: 	:	::::::::::::: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:@: 

_output_shapes
::"

_output_shapes

: 	: 

_output_shapes
:	:,(
&
_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 	

_output_shapes
:: 


_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::

_output_shapes
: 
�	
�
8__inference_batch_normalization_layer_call_fn_4374094811

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *\
fWRU
S__inference_batch_normalization_layer_call_and_return_conditional_losses_4374093412�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+���������������������������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
�	
�
1__inference_sequential_1_layer_call_fn_4374093838
conv2d_1_input!
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallconv2d_1_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *U
fPRN
L__inference_sequential_1_layer_call_and_return_conditional_losses_4374093806w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':���������	: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
/
_output_shapes
:���������	
(
_user_specified_nameconv2d_1_input
�d
�
%__inference__wrapped_model_4374093390
input_1N
4dqn_sequential_conv2d_conv2d_readvariableop_resource:C
5dqn_sequential_conv2d_biasadd_readvariableop_resource:H
:dqn_sequential_batch_normalization_readvariableop_resource:J
<dqn_sequential_batch_normalization_readvariableop_1_resource:Y
Kdqn_sequential_batch_normalization_fusedbatchnormv3_readvariableop_resource:[
Mdqn_sequential_batch_normalization_fusedbatchnormv3_readvariableop_1_resource:R
8dqn_sequential_1_conv2d_1_conv2d_readvariableop_resource:G
9dqn_sequential_1_conv2d_1_biasadd_readvariableop_resource:L
>dqn_sequential_1_batch_normalization_1_readvariableop_resource:N
@dqn_sequential_1_batch_normalization_1_readvariableop_1_resource:]
Odqn_sequential_1_batch_normalization_1_fusedbatchnormv3_readvariableop_resource:_
Qdqn_sequential_1_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource::
(dqn_dense_matmul_readvariableop_resource:@7
)dqn_dense_biasadd_readvariableop_resource::
*dqn_dense_1_matmul_readvariableop_resource:
 	9
+dqn_dense_1_biasadd_readvariableop_resource:	
identity�� dqn/dense/BiasAdd/ReadVariableOp�dqn/dense/MatMul/ReadVariableOp�"dqn/dense_1/BiasAdd/ReadVariableOp�!dqn/dense_1/MatMul/ReadVariableOp�Bdqn/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp�Ddqn/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1�1dqn/sequential/batch_normalization/ReadVariableOp�3dqn/sequential/batch_normalization/ReadVariableOp_1�,dqn/sequential/conv2d/BiasAdd/ReadVariableOp�+dqn/sequential/conv2d/Conv2D/ReadVariableOp�Fdqn/sequential_1/batch_normalization_1/FusedBatchNormV3/ReadVariableOp�Hdqn/sequential_1/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1�5dqn/sequential_1/batch_normalization_1/ReadVariableOp�7dqn/sequential_1/batch_normalization_1/ReadVariableOp_1�0dqn/sequential_1/conv2d_1/BiasAdd/ReadVariableOp�/dqn/sequential_1/conv2d_1/Conv2D/ReadVariableOp�
+dqn/sequential/conv2d/Conv2D/ReadVariableOpReadVariableOp4dqn_sequential_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
dqn/sequential/conv2d/Conv2DConv2Dinput_13dqn/sequential/conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������	*
paddingVALID*
strides
�
,dqn/sequential/conv2d/BiasAdd/ReadVariableOpReadVariableOp5dqn_sequential_conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
dqn/sequential/conv2d/BiasAddBiasAdd%dqn/sequential/conv2d/Conv2D:output:04dqn/sequential/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������	�
1dqn/sequential/batch_normalization/ReadVariableOpReadVariableOp:dqn_sequential_batch_normalization_readvariableop_resource*
_output_shapes
:*
dtype0�
3dqn/sequential/batch_normalization/ReadVariableOp_1ReadVariableOp<dqn_sequential_batch_normalization_readvariableop_1_resource*
_output_shapes
:*
dtype0�
Bdqn/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOpKdqn_sequential_batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0�
Ddqn/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpMdqn_sequential_batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0�
3dqn/sequential/batch_normalization/FusedBatchNormV3FusedBatchNormV3&dqn/sequential/conv2d/BiasAdd:output:09dqn/sequential/batch_normalization/ReadVariableOp:value:0;dqn/sequential/batch_normalization/ReadVariableOp_1:value:0Jdqn/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0Ldqn/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������	:::::*
epsilon%o�:*
is_training( �
dqn/sequential/re_lu/ReluRelu7dqn/sequential/batch_normalization/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:���������	�
/dqn/sequential_1/conv2d_1/Conv2D/ReadVariableOpReadVariableOp8dqn_sequential_1_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
 dqn/sequential_1/conv2d_1/Conv2DConv2D'dqn/sequential/re_lu/Relu:activations:07dqn/sequential_1/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
�
0dqn/sequential_1/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp9dqn_sequential_1_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
!dqn/sequential_1/conv2d_1/BiasAddBiasAdd)dqn/sequential_1/conv2d_1/Conv2D:output:08dqn/sequential_1/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:����������
5dqn/sequential_1/batch_normalization_1/ReadVariableOpReadVariableOp>dqn_sequential_1_batch_normalization_1_readvariableop_resource*
_output_shapes
:*
dtype0�
7dqn/sequential_1/batch_normalization_1/ReadVariableOp_1ReadVariableOp@dqn_sequential_1_batch_normalization_1_readvariableop_1_resource*
_output_shapes
:*
dtype0�
Fdqn/sequential_1/batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOpOdqn_sequential_1_batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0�
Hdqn/sequential_1/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpQdqn_sequential_1_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0�
7dqn/sequential_1/batch_normalization_1/FusedBatchNormV3FusedBatchNormV3*dqn/sequential_1/conv2d_1/BiasAdd:output:0=dqn/sequential_1/batch_normalization_1/ReadVariableOp:value:0?dqn/sequential_1/batch_normalization_1/ReadVariableOp_1:value:0Ndqn/sequential_1/batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0Pdqn/sequential_1/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������:::::*
epsilon%o�:*
is_training( �
dqn/sequential_1/re_lu_1/ReluRelu;dqn/sequential_1/batch_normalization_1/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:���������b
dqn/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"����@   �
dqn/flatten/ReshapeReshape+dqn/sequential_1/re_lu_1/Relu:activations:0dqn/flatten/Const:output:0*
T0*'
_output_shapes
:���������@h
dqn/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        j
dqn/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    `   j
dqn/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
dqn/strided_sliceStridedSlicedqn/flatten/Reshape:output:0 dqn/strided_slice/stack:output:0"dqn/strided_slice/stack_1:output:0"dqn/strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������@*

begin_mask*
end_maskj
dqn/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    `   l
dqn/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        l
dqn/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
dqn/strided_slice_1StridedSlicedqn/flatten/Reshape:output:0"dqn/strided_slice_1/stack:output:0$dqn/strided_slice_1/stack_1:output:0$dqn/strided_slice_1/stack_2:output:0*
Index0*
T0*%
_output_shapes
:��������� *

begin_mask*
end_mask�
dqn/dense/MatMul/ReadVariableOpReadVariableOp(dqn_dense_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0�
dqn/dense/MatMulMatMuldqn/strided_slice:output:0'dqn/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
 dqn/dense/BiasAdd/ReadVariableOpReadVariableOp)dqn_dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
dqn/dense/BiasAddBiasAdddqn/dense/MatMul:product:0(dqn/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
!dqn/dense_1/MatMul/ReadVariableOpReadVariableOp*dqn_dense_1_matmul_readvariableop_resource*
_output_shapes

: 	*
dtype0�
dqn/dense_1/MatMulMatMuldqn/strided_slice_1:output:0)dqn/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������	�
"dqn/dense_1/BiasAdd/ReadVariableOpReadVariableOp+dqn_dense_1_biasadd_readvariableop_resource*
_output_shapes
:	*
dtype0�
dqn/dense_1/BiasAddBiasAdddqn/dense_1/MatMul:product:0*dqn/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������	\
dqn/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :�
dqn/MeanMeandqn/dense_1/BiasAdd:output:0#dqn/Mean/reduction_indices:output:0*
T0*#
_output_shapes
:���������b
dqn/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"����   w
dqn/ReshapeReshapedqn/Mean:output:0dqn/Reshape/shape:output:0*
T0*'
_output_shapes
:���������t
dqn/subSubdqn/dense_1/BiasAdd:output:0dqn/Reshape:output:0*
T0*'
_output_shapes
:���������	k
dqn/addAddV2dqn/dense/BiasAdd:output:0dqn/sub:z:0*
T0*'
_output_shapes
:���������	Z
IdentityIdentitydqn/add:z:0^NoOp*
T0*'
_output_shapes
:���������	�
NoOpNoOp!^dqn/dense/BiasAdd/ReadVariableOp ^dqn/dense/MatMul/ReadVariableOp#^dqn/dense_1/BiasAdd/ReadVariableOp"^dqn/dense_1/MatMul/ReadVariableOpC^dqn/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOpE^dqn/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_12^dqn/sequential/batch_normalization/ReadVariableOp4^dqn/sequential/batch_normalization/ReadVariableOp_1-^dqn/sequential/conv2d/BiasAdd/ReadVariableOp,^dqn/sequential/conv2d/Conv2D/ReadVariableOpG^dqn/sequential_1/batch_normalization_1/FusedBatchNormV3/ReadVariableOpI^dqn/sequential_1/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_16^dqn/sequential_1/batch_normalization_1/ReadVariableOp8^dqn/sequential_1/batch_normalization_1/ReadVariableOp_11^dqn/sequential_1/conv2d_1/BiasAdd/ReadVariableOp0^dqn/sequential_1/conv2d_1/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:���������: : : : : : : : : : : : : : : : 2D
 dqn/dense/BiasAdd/ReadVariableOp dqn/dense/BiasAdd/ReadVariableOp2B
dqn/dense/MatMul/ReadVariableOpdqn/dense/MatMul/ReadVariableOp2H
"dqn/dense_1/BiasAdd/ReadVariableOp"dqn/dense_1/BiasAdd/ReadVariableOp2F
!dqn/dense_1/MatMul/ReadVariableOp!dqn/dense_1/MatMul/ReadVariableOp2�
Bdqn/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOpBdqn/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp2�
Ddqn/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1Ddqn/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_12f
1dqn/sequential/batch_normalization/ReadVariableOp1dqn/sequential/batch_normalization/ReadVariableOp2j
3dqn/sequential/batch_normalization/ReadVariableOp_13dqn/sequential/batch_normalization/ReadVariableOp_12\
,dqn/sequential/conv2d/BiasAdd/ReadVariableOp,dqn/sequential/conv2d/BiasAdd/ReadVariableOp2Z
+dqn/sequential/conv2d/Conv2D/ReadVariableOp+dqn/sequential/conv2d/Conv2D/ReadVariableOp2�
Fdqn/sequential_1/batch_normalization_1/FusedBatchNormV3/ReadVariableOpFdqn/sequential_1/batch_normalization_1/FusedBatchNormV3/ReadVariableOp2�
Hdqn/sequential_1/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1Hdqn/sequential_1/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_12n
5dqn/sequential_1/batch_normalization_1/ReadVariableOp5dqn/sequential_1/batch_normalization_1/ReadVariableOp2r
7dqn/sequential_1/batch_normalization_1/ReadVariableOp_17dqn/sequential_1/batch_normalization_1/ReadVariableOp_12d
0dqn/sequential_1/conv2d_1/BiasAdd/ReadVariableOp0dqn/sequential_1/conv2d_1/BiasAdd/ReadVariableOp2b
/dqn/sequential_1/conv2d_1/Conv2D/ReadVariableOp/dqn/sequential_1/conv2d_1/Conv2D/ReadVariableOp:X T
/
_output_shapes
:���������
!
_user_specified_name	input_1
�
�
J__inference_sequential_layer_call_and_return_conditional_losses_4374093633
conv2d_input+
conv2d_4374093617:
conv2d_4374093619:,
batch_normalization_4374093622:,
batch_normalization_4374093624:,
batch_normalization_4374093626:,
batch_normalization_4374093628:
identity��+batch_normalization/StatefulPartitionedCall�conv2d/StatefulPartitionedCall�
conv2d/StatefulPartitionedCallStatefulPartitionedCallconv2d_inputconv2d_4374093617conv2d_4374093619*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������	*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_conv2d_layer_call_and_return_conditional_losses_4374093471�
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0batch_normalization_4374093622batch_normalization_4374093624batch_normalization_4374093626batch_normalization_4374093628*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������	*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *\
fWRU
S__inference_batch_normalization_layer_call_and_return_conditional_losses_4374093443�
re_lu/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������	* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_re_lu_layer_call_and_return_conditional_losses_4374093491u
IdentityIdentityre_lu/PartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������	�
NoOpNoOp,^batch_normalization/StatefulPartitionedCall^conv2d/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':���������: : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall:] Y
/
_output_shapes
:���������
&
_user_specified_nameconv2d_input
�	
�
1__inference_sequential_1_layer_call_fn_4374094680

inputs!
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *U
fPRN
L__inference_sequential_1_layer_call_and_return_conditional_losses_4374093806w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':���������	: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������	
 
_user_specified_nameinputs
�
c
G__inference_flatten_layer_call_and_return_conditional_losses_4374094741

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"����@   \
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:���������@X
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:���������@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
J__inference_sequential_layer_call_and_return_conditional_losses_4374093614
conv2d_input+
conv2d_4374093598:
conv2d_4374093600:,
batch_normalization_4374093603:,
batch_normalization_4374093605:,
batch_normalization_4374093607:,
batch_normalization_4374093609:
identity��+batch_normalization/StatefulPartitionedCall�conv2d/StatefulPartitionedCall�
conv2d/StatefulPartitionedCallStatefulPartitionedCallconv2d_inputconv2d_4374093598conv2d_4374093600*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������	*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_conv2d_layer_call_and_return_conditional_losses_4374093471�
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0batch_normalization_4374093603batch_normalization_4374093605batch_normalization_4374093607batch_normalization_4374093609*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������	*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *\
fWRU
S__inference_batch_normalization_layer_call_and_return_conditional_losses_4374093412�
re_lu/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������	* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_re_lu_layer_call_and_return_conditional_losses_4374093491u
IdentityIdentityre_lu/PartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������	�
NoOpNoOp,^batch_normalization/StatefulPartitionedCall^conv2d/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':���������: : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall:] Y
/
_output_shapes
:���������
&
_user_specified_nameconv2d_input
�n
�
C__inference_dqn_layer_call_and_return_conditional_losses_4374094523
xJ
0sequential_conv2d_conv2d_readvariableop_resource:?
1sequential_conv2d_biasadd_readvariableop_resource:D
6sequential_batch_normalization_readvariableop_resource:F
8sequential_batch_normalization_readvariableop_1_resource:U
Gsequential_batch_normalization_fusedbatchnormv3_readvariableop_resource:W
Isequential_batch_normalization_fusedbatchnormv3_readvariableop_1_resource:N
4sequential_1_conv2d_1_conv2d_readvariableop_resource:C
5sequential_1_conv2d_1_biasadd_readvariableop_resource:H
:sequential_1_batch_normalization_1_readvariableop_resource:J
<sequential_1_batch_normalization_1_readvariableop_1_resource:Y
Ksequential_1_batch_normalization_1_fusedbatchnormv3_readvariableop_resource:[
Msequential_1_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource:6
$dense_matmul_readvariableop_resource:@3
%dense_biasadd_readvariableop_resource:6
&dense_1_matmul_readvariableop_resource:
 	5
'dense_1_biasadd_readvariableop_resource:	
identity��dense/BiasAdd/ReadVariableOp�dense/MatMul/ReadVariableOp�dense_1/BiasAdd/ReadVariableOp�dense_1/MatMul/ReadVariableOp�-sequential/batch_normalization/AssignNewValue�/sequential/batch_normalization/AssignNewValue_1�>sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp�@sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1�-sequential/batch_normalization/ReadVariableOp�/sequential/batch_normalization/ReadVariableOp_1�(sequential/conv2d/BiasAdd/ReadVariableOp�'sequential/conv2d/Conv2D/ReadVariableOp�1sequential_1/batch_normalization_1/AssignNewValue�3sequential_1/batch_normalization_1/AssignNewValue_1�Bsequential_1/batch_normalization_1/FusedBatchNormV3/ReadVariableOp�Dsequential_1/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1�1sequential_1/batch_normalization_1/ReadVariableOp�3sequential_1/batch_normalization_1/ReadVariableOp_1�,sequential_1/conv2d_1/BiasAdd/ReadVariableOp�+sequential_1/conv2d_1/Conv2D/ReadVariableOp�
'sequential/conv2d/Conv2D/ReadVariableOpReadVariableOp0sequential_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
sequential/conv2d/Conv2DConv2Dx/sequential/conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������	*
paddingVALID*
strides
�
(sequential/conv2d/BiasAdd/ReadVariableOpReadVariableOp1sequential_conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
sequential/conv2d/BiasAddBiasAdd!sequential/conv2d/Conv2D:output:00sequential/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������	�
-sequential/batch_normalization/ReadVariableOpReadVariableOp6sequential_batch_normalization_readvariableop_resource*
_output_shapes
:*
dtype0�
/sequential/batch_normalization/ReadVariableOp_1ReadVariableOp8sequential_batch_normalization_readvariableop_1_resource*
_output_shapes
:*
dtype0�
>sequential/batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOpGsequential_batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0�
@sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpIsequential_batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0�
/sequential/batch_normalization/FusedBatchNormV3FusedBatchNormV3"sequential/conv2d/BiasAdd:output:05sequential/batch_normalization/ReadVariableOp:value:07sequential/batch_normalization/ReadVariableOp_1:value:0Fsequential/batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0Hsequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������	:::::*
epsilon%o�:*
exponential_avg_factor%
�#<�
-sequential/batch_normalization/AssignNewValueAssignVariableOpGsequential_batch_normalization_fusedbatchnormv3_readvariableop_resource<sequential/batch_normalization/FusedBatchNormV3:batch_mean:0?^sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0�
/sequential/batch_normalization/AssignNewValue_1AssignVariableOpIsequential_batch_normalization_fusedbatchnormv3_readvariableop_1_resource@sequential/batch_normalization/FusedBatchNormV3:batch_variance:0A^sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0�
sequential/re_lu/ReluRelu3sequential/batch_normalization/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:���������	�
+sequential_1/conv2d_1/Conv2D/ReadVariableOpReadVariableOp4sequential_1_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
sequential_1/conv2d_1/Conv2DConv2D#sequential/re_lu/Relu:activations:03sequential_1/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
�
,sequential_1/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp5sequential_1_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
sequential_1/conv2d_1/BiasAddBiasAdd%sequential_1/conv2d_1/Conv2D:output:04sequential_1/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:����������
1sequential_1/batch_normalization_1/ReadVariableOpReadVariableOp:sequential_1_batch_normalization_1_readvariableop_resource*
_output_shapes
:*
dtype0�
3sequential_1/batch_normalization_1/ReadVariableOp_1ReadVariableOp<sequential_1_batch_normalization_1_readvariableop_1_resource*
_output_shapes
:*
dtype0�
Bsequential_1/batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOpKsequential_1_batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0�
Dsequential_1/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpMsequential_1_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0�
3sequential_1/batch_normalization_1/FusedBatchNormV3FusedBatchNormV3&sequential_1/conv2d_1/BiasAdd:output:09sequential_1/batch_normalization_1/ReadVariableOp:value:0;sequential_1/batch_normalization_1/ReadVariableOp_1:value:0Jsequential_1/batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0Lsequential_1/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������:::::*
epsilon%o�:*
exponential_avg_factor%
�#<�
1sequential_1/batch_normalization_1/AssignNewValueAssignVariableOpKsequential_1_batch_normalization_1_fusedbatchnormv3_readvariableop_resource@sequential_1/batch_normalization_1/FusedBatchNormV3:batch_mean:0C^sequential_1/batch_normalization_1/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0�
3sequential_1/batch_normalization_1/AssignNewValue_1AssignVariableOpMsequential_1_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resourceDsequential_1/batch_normalization_1/FusedBatchNormV3:batch_variance:0E^sequential_1/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0�
sequential_1/re_lu_1/ReluRelu7sequential_1/batch_normalization_1/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:���������^
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"����@   �
flatten/ReshapeReshape'sequential_1/re_lu_1/Relu:activations:0flatten/Const:output:0*
T0*'
_output_shapes
:���������@d
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        f
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    `   f
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
strided_sliceStridedSliceflatten/Reshape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������@*

begin_mask*
end_maskf
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    `   h
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        h
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
strided_slice_1StridedSliceflatten/Reshape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*%
_output_shapes
:��������� *

begin_mask*
end_mask�
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0�
dense/MatMulMatMulstrided_slice:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

: 	*
dtype0�
dense_1/MatMulMatMulstrided_slice_1:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������	�
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:	*
dtype0�
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������	X
Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :u
MeanMeandense_1/BiasAdd:output:0Mean/reduction_indices:output:0*
T0*#
_output_shapes
:���������^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"����   k
ReshapeReshapeMean:output:0Reshape/shape:output:0*
T0*'
_output_shapes
:���������h
subSubdense_1/BiasAdd:output:0Reshape:output:0*
T0*'
_output_shapes
:���������	_
addAddV2dense/BiasAdd:output:0sub:z:0*
T0*'
_output_shapes
:���������	V
IdentityIdentityadd:z:0^NoOp*
T0*'
_output_shapes
:���������	�
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp.^sequential/batch_normalization/AssignNewValue0^sequential/batch_normalization/AssignNewValue_1?^sequential/batch_normalization/FusedBatchNormV3/ReadVariableOpA^sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1.^sequential/batch_normalization/ReadVariableOp0^sequential/batch_normalization/ReadVariableOp_1)^sequential/conv2d/BiasAdd/ReadVariableOp(^sequential/conv2d/Conv2D/ReadVariableOp2^sequential_1/batch_normalization_1/AssignNewValue4^sequential_1/batch_normalization_1/AssignNewValue_1C^sequential_1/batch_normalization_1/FusedBatchNormV3/ReadVariableOpE^sequential_1/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_12^sequential_1/batch_normalization_1/ReadVariableOp4^sequential_1/batch_normalization_1/ReadVariableOp_1-^sequential_1/conv2d_1/BiasAdd/ReadVariableOp,^sequential_1/conv2d_1/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:���������: : : : : : : : : : : : : : : : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2^
-sequential/batch_normalization/AssignNewValue-sequential/batch_normalization/AssignNewValue2b
/sequential/batch_normalization/AssignNewValue_1/sequential/batch_normalization/AssignNewValue_12�
>sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp>sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp2�
@sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1@sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_12^
-sequential/batch_normalization/ReadVariableOp-sequential/batch_normalization/ReadVariableOp2b
/sequential/batch_normalization/ReadVariableOp_1/sequential/batch_normalization/ReadVariableOp_12T
(sequential/conv2d/BiasAdd/ReadVariableOp(sequential/conv2d/BiasAdd/ReadVariableOp2R
'sequential/conv2d/Conv2D/ReadVariableOp'sequential/conv2d/Conv2D/ReadVariableOp2f
1sequential_1/batch_normalization_1/AssignNewValue1sequential_1/batch_normalization_1/AssignNewValue2j
3sequential_1/batch_normalization_1/AssignNewValue_13sequential_1/batch_normalization_1/AssignNewValue_12�
Bsequential_1/batch_normalization_1/FusedBatchNormV3/ReadVariableOpBsequential_1/batch_normalization_1/FusedBatchNormV3/ReadVariableOp2�
Dsequential_1/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1Dsequential_1/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_12f
1sequential_1/batch_normalization_1/ReadVariableOp1sequential_1/batch_normalization_1/ReadVariableOp2j
3sequential_1/batch_normalization_1/ReadVariableOp_13sequential_1/batch_normalization_1/ReadVariableOp_12\
,sequential_1/conv2d_1/BiasAdd/ReadVariableOp,sequential_1/conv2d_1/BiasAdd/ReadVariableOp2Z
+sequential_1/conv2d_1/Conv2D/ReadVariableOp+sequential_1/conv2d_1/Conv2D/ReadVariableOp:R N
/
_output_shapes
:���������

_user_specified_namex
�
a
E__inference_re_lu_layer_call_and_return_conditional_losses_4374093491

inputs
identityN
ReluReluinputs*
T0*/
_output_shapes
:���������	b
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:���������	"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������	:W S
/
_output_shapes
:���������	
 
_user_specified_nameinputs
�
F
*__inference_re_lu_layer_call_fn_4374094865

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������	* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_re_lu_layer_call_and_return_conditional_losses_4374093491h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:���������	"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������	:W S
/
_output_shapes
:���������	
 
_user_specified_nameinputs
�
�
(__inference_signature_wrapper_4374094562
input_1!
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:#
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:

unknown_11:@

unknown_12:

unknown_13:
 	

unknown_14:	
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������	*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *.
f)R'
%__inference__wrapped_model_4374093390o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������	`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:���������: : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:���������
!
_user_specified_name	input_1
�

�
H__inference_conv2d_1_layer_call_and_return_conditional_losses_4374093714

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������	: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������	
 
_user_specified_nameinputs
�
�
(__inference_dqn_layer_call_fn_4374094191
input_1!
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:#
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:

unknown_11:@

unknown_12:

unknown_13:
 	

unknown_14:	
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������	*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dqn_layer_call_and_return_conditional_losses_4374094119o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������	`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:���������: : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:���������
!
_user_specified_name	input_1
�
�
L__inference_sequential_1_layer_call_and_return_conditional_losses_4374093737

inputs-
conv2d_1_4374093715:!
conv2d_1_4374093717:.
 batch_normalization_1_4374093720:.
 batch_normalization_1_4374093722:.
 batch_normalization_1_4374093724:.
 batch_normalization_1_4374093726:
identity��-batch_normalization_1/StatefulPartitionedCall� conv2d_1/StatefulPartitionedCall�
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_1_4374093715conv2d_1_4374093717*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_conv2d_1_layer_call_and_return_conditional_losses_4374093714�
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0 batch_normalization_1_4374093720 batch_normalization_1_4374093722 batch_normalization_1_4374093724 batch_normalization_1_4374093726*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *^
fYRW
U__inference_batch_normalization_1_layer_call_and_return_conditional_losses_4374093655�
re_lu_1/PartitionedCallPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_re_lu_1_layer_call_and_return_conditional_losses_4374093734w
IdentityIdentity re_lu_1/PartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:����������
NoOpNoOp.^batch_normalization_1/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':���������	: : : : : : 2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall:W S
/
_output_shapes
:���������	
 
_user_specified_nameinputs
�	
�
E__inference_dense_layer_call_and_return_conditional_losses_4374093935

inputs0
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
J__inference_sequential_layer_call_and_return_conditional_losses_4374094621

inputs?
%conv2d_conv2d_readvariableop_resource:4
&conv2d_biasadd_readvariableop_resource:9
+batch_normalization_readvariableop_resource:;
-batch_normalization_readvariableop_1_resource:J
<batch_normalization_fusedbatchnormv3_readvariableop_resource:L
>batch_normalization_fusedbatchnormv3_readvariableop_1_resource:
identity��3batch_normalization/FusedBatchNormV3/ReadVariableOp�5batch_normalization/FusedBatchNormV3/ReadVariableOp_1�"batch_normalization/ReadVariableOp�$batch_normalization/ReadVariableOp_1�conv2d/BiasAdd/ReadVariableOp�conv2d/Conv2D/ReadVariableOp�
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
conv2d/Conv2DConv2Dinputs$conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������	*
paddingVALID*
strides
�
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������	�
"batch_normalization/ReadVariableOpReadVariableOp+batch_normalization_readvariableop_resource*
_output_shapes
:*
dtype0�
$batch_normalization/ReadVariableOp_1ReadVariableOp-batch_normalization_readvariableop_1_resource*
_output_shapes
:*
dtype0�
3batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOp<batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0�
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp>batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0�
$batch_normalization/FusedBatchNormV3FusedBatchNormV3conv2d/BiasAdd:output:0*batch_normalization/ReadVariableOp:value:0,batch_normalization/ReadVariableOp_1:value:0;batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0=batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������	:::::*
epsilon%o�:*
is_training( v

re_lu/ReluRelu(batch_normalization/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:���������	o
IdentityIdentityre_lu/Relu:activations:0^NoOp*
T0*/
_output_shapes
:���������	�
NoOpNoOp4^batch_normalization/FusedBatchNormV3/ReadVariableOp6^batch_normalization/FusedBatchNormV3/ReadVariableOp_1#^batch_normalization/ReadVariableOp%^batch_normalization/ReadVariableOp_1^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':���������: : : : : : 2j
3batch_normalization/FusedBatchNormV3/ReadVariableOp3batch_normalization/FusedBatchNormV3/ReadVariableOp2n
5batch_normalization/FusedBatchNormV3/ReadVariableOp_15batch_normalization/FusedBatchNormV3/ReadVariableOp_12H
"batch_normalization/ReadVariableOp"batch_normalization/ReadVariableOp2L
$batch_normalization/ReadVariableOp_1$batch_normalization/ReadVariableOp_12>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�	
�
G__inference_dense_1_layer_call_and_return_conditional_losses_4374093951

inputs.
matmul_readvariableop_resource:
 	-
biasadd_readvariableop_resource:	
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpr
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: 	*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������	r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:	*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������	_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������	w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:��������� : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:M I
%
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
�
S__inference_batch_normalization_layer_call_and_return_conditional_losses_4374093412

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������:::::*
epsilon%o�:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+����������������������������
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+���������������������������: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
�
c
G__inference_flatten_layer_call_and_return_conditional_losses_4374093915

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"����@   \
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:���������@X
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:���������@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
S__inference_batch_normalization_layer_call_and_return_conditional_losses_4374094860

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������:::::*
epsilon%o�:*
exponential_avg_factor%
�#<�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+����������������������������
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+���������������������������: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
�	
�
8__inference_batch_normalization_layer_call_fn_4374094824

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *\
fWRU
S__inference_batch_normalization_layer_call_and_return_conditional_losses_4374093443�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+���������������������������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
�
�
(__inference_dqn_layer_call_fn_4374094375
x!
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:#
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:

unknown_11:@

unknown_12:

unknown_13:
 	

unknown_14:	
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallxunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������	*.
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dqn_layer_call_and_return_conditional_losses_4374094119o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������	`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:���������: : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:R N
/
_output_shapes
:���������

_user_specified_namex
�	
�
1__inference_sequential_1_layer_call_fn_4374094663

inputs!
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *U
fPRN
L__inference_sequential_1_layer_call_and_return_conditional_losses_4374093737w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':���������	: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������	
 
_user_specified_nameinputs
�
c
G__inference_re_lu_1_layer_call_and_return_conditional_losses_4374094961

inputs
identityN
ReluReluinputs*
T0*/
_output_shapes
:���������b
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
+__inference_conv2d_layer_call_fn_4374094788

inputs!
unknown:
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������	*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_conv2d_layer_call_and_return_conditional_losses_4374093471w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������	`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�	
�
1__inference_sequential_1_layer_call_fn_4374093752
conv2d_1_input!
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallconv2d_1_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *U
fPRN
L__inference_sequential_1_layer_call_and_return_conditional_losses_4374093737w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':���������	: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
/
_output_shapes
:���������	
(
_user_specified_nameconv2d_1_input
�
�
*__inference_dense_layer_call_fn_4374094750

inputs
unknown:@
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dense_layer_call_and_return_conditional_losses_4374093935o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������@: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�	
�
/__inference_sequential_layer_call_fn_4374094579

inputs!
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������	*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *S
fNRL
J__inference_sequential_layer_call_and_return_conditional_losses_4374093494w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������	`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':���������: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�	
�
E__inference_dense_layer_call_and_return_conditional_losses_4374094760

inputs0
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
H
,__inference_flatten_layer_call_fn_4374094735

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_flatten_layer_call_and_return_conditional_losses_4374093915`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:���������@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
(__inference_dqn_layer_call_fn_4374094338
x!
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:#
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:

unknown_11:@

unknown_12:

unknown_13:
 	

unknown_14:	
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallxunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������	*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dqn_layer_call_and_return_conditional_losses_4374093964o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������	`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:���������: : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:R N
/
_output_shapes
:���������

_user_specified_namex
�
�
,__inference_dense_1_layer_call_fn_4374094769

inputs
unknown:
 	
	unknown_0:	
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������	*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_dense_1_layer_call_and_return_conditional_losses_4374093951o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������	`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:��������� : : 22
StatefulPartitionedCallStatefulPartitionedCall:M I
%
_output_shapes
:��������� 
 
_user_specified_nameinputs
�#
�
J__inference_sequential_layer_call_and_return_conditional_losses_4374094646

inputs?
%conv2d_conv2d_readvariableop_resource:4
&conv2d_biasadd_readvariableop_resource:9
+batch_normalization_readvariableop_resource:;
-batch_normalization_readvariableop_1_resource:J
<batch_normalization_fusedbatchnormv3_readvariableop_resource:L
>batch_normalization_fusedbatchnormv3_readvariableop_1_resource:
identity��"batch_normalization/AssignNewValue�$batch_normalization/AssignNewValue_1�3batch_normalization/FusedBatchNormV3/ReadVariableOp�5batch_normalization/FusedBatchNormV3/ReadVariableOp_1�"batch_normalization/ReadVariableOp�$batch_normalization/ReadVariableOp_1�conv2d/BiasAdd/ReadVariableOp�conv2d/Conv2D/ReadVariableOp�
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
conv2d/Conv2DConv2Dinputs$conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������	*
paddingVALID*
strides
�
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������	�
"batch_normalization/ReadVariableOpReadVariableOp+batch_normalization_readvariableop_resource*
_output_shapes
:*
dtype0�
$batch_normalization/ReadVariableOp_1ReadVariableOp-batch_normalization_readvariableop_1_resource*
_output_shapes
:*
dtype0�
3batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOp<batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0�
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp>batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0�
$batch_normalization/FusedBatchNormV3FusedBatchNormV3conv2d/BiasAdd:output:0*batch_normalization/ReadVariableOp:value:0,batch_normalization/ReadVariableOp_1:value:0;batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0=batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������	:::::*
epsilon%o�:*
exponential_avg_factor%
�#<�
"batch_normalization/AssignNewValueAssignVariableOp<batch_normalization_fusedbatchnormv3_readvariableop_resource1batch_normalization/FusedBatchNormV3:batch_mean:04^batch_normalization/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0�
$batch_normalization/AssignNewValue_1AssignVariableOp>batch_normalization_fusedbatchnormv3_readvariableop_1_resource5batch_normalization/FusedBatchNormV3:batch_variance:06^batch_normalization/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0v

re_lu/ReluRelu(batch_normalization/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:���������	o
IdentityIdentityre_lu/Relu:activations:0^NoOp*
T0*/
_output_shapes
:���������	�
NoOpNoOp#^batch_normalization/AssignNewValue%^batch_normalization/AssignNewValue_14^batch_normalization/FusedBatchNormV3/ReadVariableOp6^batch_normalization/FusedBatchNormV3/ReadVariableOp_1#^batch_normalization/ReadVariableOp%^batch_normalization/ReadVariableOp_1^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':���������: : : : : : 2H
"batch_normalization/AssignNewValue"batch_normalization/AssignNewValue2L
$batch_normalization/AssignNewValue_1$batch_normalization/AssignNewValue_12j
3batch_normalization/FusedBatchNormV3/ReadVariableOp3batch_normalization/FusedBatchNormV3/ReadVariableOp2n
5batch_normalization/FusedBatchNormV3/ReadVariableOp_15batch_normalization/FusedBatchNormV3/ReadVariableOp_12H
"batch_normalization/ReadVariableOp"batch_normalization/ReadVariableOp2L
$batch_normalization/ReadVariableOp_1$batch_normalization/ReadVariableOp_12>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
(__inference_dqn_layer_call_fn_4374093999
input_1!
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:#
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:

unknown_11:@

unknown_12:

unknown_13:
 	

unknown_14:	
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������	*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_dqn_layer_call_and_return_conditional_losses_4374093964o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������	`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:���������: : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:���������
!
_user_specified_name	input_1
�
�
L__inference_sequential_1_layer_call_and_return_conditional_losses_4374093857
conv2d_1_input-
conv2d_1_4374093841:!
conv2d_1_4374093843:.
 batch_normalization_1_4374093846:.
 batch_normalization_1_4374093848:.
 batch_normalization_1_4374093850:.
 batch_normalization_1_4374093852:
identity��-batch_normalization_1/StatefulPartitionedCall� conv2d_1/StatefulPartitionedCall�
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCallconv2d_1_inputconv2d_1_4374093841conv2d_1_4374093843*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_conv2d_1_layer_call_and_return_conditional_losses_4374093714�
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0 batch_normalization_1_4374093846 batch_normalization_1_4374093848 batch_normalization_1_4374093850 batch_normalization_1_4374093852*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *^
fYRW
U__inference_batch_normalization_1_layer_call_and_return_conditional_losses_4374093655�
re_lu_1/PartitionedCallPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_re_lu_1_layer_call_and_return_conditional_losses_4374093734w
IdentityIdentity re_lu_1/PartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:����������
NoOpNoOp.^batch_normalization_1/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':���������	: : : : : : 2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall:_ [
/
_output_shapes
:���������	
(
_user_specified_nameconv2d_1_input
�
�
L__inference_sequential_1_layer_call_and_return_conditional_losses_4374093876
conv2d_1_input-
conv2d_1_4374093860:!
conv2d_1_4374093862:.
 batch_normalization_1_4374093865:.
 batch_normalization_1_4374093867:.
 batch_normalization_1_4374093869:.
 batch_normalization_1_4374093871:
identity��-batch_normalization_1/StatefulPartitionedCall� conv2d_1/StatefulPartitionedCall�
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCallconv2d_1_inputconv2d_1_4374093860conv2d_1_4374093862*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_conv2d_1_layer_call_and_return_conditional_losses_4374093714�
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0 batch_normalization_1_4374093865 batch_normalization_1_4374093867 batch_normalization_1_4374093869 batch_normalization_1_4374093871*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *^
fYRW
U__inference_batch_normalization_1_layer_call_and_return_conditional_losses_4374093686�
re_lu_1/PartitionedCallPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_re_lu_1_layer_call_and_return_conditional_losses_4374093734w
IdentityIdentity re_lu_1/PartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:����������
NoOpNoOp.^batch_normalization_1/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':���������	: : : : : : 2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall:_ [
/
_output_shapes
:���������	
(
_user_specified_nameconv2d_1_input
�
�
U__inference_batch_normalization_1_layer_call_and_return_conditional_losses_4374094951

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������:::::*
epsilon%o�:*
exponential_avg_factor%
�#<�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+����������������������������
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+���������������������������: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
�
�
J__inference_sequential_layer_call_and_return_conditional_losses_4374093494

inputs+
conv2d_4374093472:
conv2d_4374093474:,
batch_normalization_4374093477:,
batch_normalization_4374093479:,
batch_normalization_4374093481:,
batch_normalization_4374093483:
identity��+batch_normalization/StatefulPartitionedCall�conv2d/StatefulPartitionedCall�
conv2d/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_4374093472conv2d_4374093474*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������	*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_conv2d_layer_call_and_return_conditional_losses_4374093471�
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0batch_normalization_4374093477batch_normalization_4374093479batch_normalization_4374093481batch_normalization_4374093483*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������	*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *\
fWRU
S__inference_batch_normalization_layer_call_and_return_conditional_losses_4374093412�
re_lu/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������	* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_re_lu_layer_call_and_return_conditional_losses_4374093491u
IdentityIdentityre_lu/PartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������	�
NoOpNoOp,^batch_normalization/StatefulPartitionedCall^conv2d/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':���������: : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
H
,__inference_re_lu_1_layer_call_fn_4374094956

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_re_lu_1_layer_call_and_return_conditional_losses_4374093734h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�	
�
/__inference_sequential_layer_call_fn_4374093595
conv2d_input!
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallconv2d_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������	*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *S
fNRL
J__inference_sequential_layer_call_and_return_conditional_losses_4374093563w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������	`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':���������: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
/
_output_shapes
:���������
&
_user_specified_nameconv2d_input
�
�
L__inference_sequential_1_layer_call_and_return_conditional_losses_4374094705

inputsA
'conv2d_1_conv2d_readvariableop_resource:6
(conv2d_1_biasadd_readvariableop_resource:;
-batch_normalization_1_readvariableop_resource:=
/batch_normalization_1_readvariableop_1_resource:L
>batch_normalization_1_fusedbatchnormv3_readvariableop_resource:N
@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource:
identity��5batch_normalization_1/FusedBatchNormV3/ReadVariableOp�7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1�$batch_normalization_1/ReadVariableOp�&batch_normalization_1/ReadVariableOp_1�conv2d_1/BiasAdd/ReadVariableOp�conv2d_1/Conv2D/ReadVariableOp�
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
conv2d_1/Conv2DConv2Dinputs&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
�
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:����������
$batch_normalization_1/ReadVariableOpReadVariableOp-batch_normalization_1_readvariableop_resource*
_output_shapes
:*
dtype0�
&batch_normalization_1/ReadVariableOp_1ReadVariableOp/batch_normalization_1_readvariableop_1_resource*
_output_shapes
:*
dtype0�
5batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0�
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0�
&batch_normalization_1/FusedBatchNormV3FusedBatchNormV3conv2d_1/BiasAdd:output:0,batch_normalization_1/ReadVariableOp:value:0.batch_normalization_1/ReadVariableOp_1:value:0=batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������:::::*
epsilon%o�:*
is_training( z
re_lu_1/ReluRelu*batch_normalization_1/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:���������q
IdentityIdentityre_lu_1/Relu:activations:0^NoOp*
T0*/
_output_shapes
:����������
NoOpNoOp6^batch_normalization_1/FusedBatchNormV3/ReadVariableOp8^batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_1/ReadVariableOp'^batch_normalization_1/ReadVariableOp_1 ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':���������	: : : : : : 2n
5batch_normalization_1/FusedBatchNormV3/ReadVariableOp5batch_normalization_1/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_17batch_normalization_1/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_1/ReadVariableOp$batch_normalization_1/ReadVariableOp2P
&batch_normalization_1/ReadVariableOp_1&batch_normalization_1/ReadVariableOp_12B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp:W S
/
_output_shapes
:���������	
 
_user_specified_nameinputs
�
�
S__inference_batch_normalization_layer_call_and_return_conditional_losses_4374094842

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������:::::*
epsilon%o�:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+����������������������������
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+���������������������������: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
�
a
E__inference_re_lu_layer_call_and_return_conditional_losses_4374094870

inputs
identityN
ReluReluinputs*
T0*/
_output_shapes
:���������	b
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:���������	"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������	:W S
/
_output_shapes
:���������	
 
_user_specified_nameinputs
�,
�
C__inference_dqn_layer_call_and_return_conditional_losses_4374093964
x/
sequential_4374093883:#
sequential_4374093885:#
sequential_4374093887:#
sequential_4374093889:#
sequential_4374093891:#
sequential_4374093893:1
sequential_1_4374093896:%
sequential_1_4374093898:%
sequential_1_4374093900:%
sequential_1_4374093902:%
sequential_1_4374093904:%
sequential_1_4374093906:"
dense_4374093936:@
dense_4374093938:"
dense_1_4374093952:
 	 
dense_1_4374093954:	
identity��dense/StatefulPartitionedCall�dense_1/StatefulPartitionedCall�"sequential/StatefulPartitionedCall�$sequential_1/StatefulPartitionedCall�
"sequential/StatefulPartitionedCallStatefulPartitionedCallxsequential_4374093883sequential_4374093885sequential_4374093887sequential_4374093889sequential_4374093891sequential_4374093893*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������	*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *S
fNRL
J__inference_sequential_layer_call_and_return_conditional_losses_4374093494�
$sequential_1/StatefulPartitionedCallStatefulPartitionedCall+sequential/StatefulPartitionedCall:output:0sequential_1_4374093896sequential_1_4374093898sequential_1_4374093900sequential_1_4374093902sequential_1_4374093904sequential_1_4374093906*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *U
fPRN
L__inference_sequential_1_layer_call_and_return_conditional_losses_4374093737�
flatten/PartitionedCallPartitionedCall-sequential_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_flatten_layer_call_and_return_conditional_losses_4374093915d
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        f
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    `   f
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
strided_sliceStridedSlice flatten/PartitionedCall:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������@*

begin_mask*
end_maskf
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    `   h
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        h
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
strided_slice_1StridedSlice flatten/PartitionedCall:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*%
_output_shapes
:��������� *

begin_mask*
end_mask�
dense/StatefulPartitionedCallStatefulPartitionedCallstrided_slice:output:0dense_4374093936dense_4374093938*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dense_layer_call_and_return_conditional_losses_4374093935�
dense_1/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_1:output:0dense_1_4374093952dense_1_4374093954*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������	*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_dense_1_layer_call_and_return_conditional_losses_4374093951X
Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :�
MeanMean(dense_1/StatefulPartitionedCall:output:0Mean/reduction_indices:output:0*
T0*#
_output_shapes
:���������^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"����   k
ReshapeReshapeMean:output:0Reshape/shape:output:0*
T0*'
_output_shapes
:���������x
subSub(dense_1/StatefulPartitionedCall:output:0Reshape:output:0*
T0*'
_output_shapes
:���������	o
addAddV2&dense/StatefulPartitionedCall:output:0sub:z:0*
T0*'
_output_shapes
:���������	V
IdentityIdentityadd:z:0^NoOp*
T0*'
_output_shapes
:���������	�
NoOpNoOp^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall#^sequential/StatefulPartitionedCall%^sequential_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:���������: : : : : : : : : : : : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2H
"sequential/StatefulPartitionedCall"sequential/StatefulPartitionedCall2L
$sequential_1/StatefulPartitionedCall$sequential_1/StatefulPartitionedCall:R N
/
_output_shapes
:���������

_user_specified_namex
�
�
L__inference_sequential_1_layer_call_and_return_conditional_losses_4374093806

inputs-
conv2d_1_4374093790:!
conv2d_1_4374093792:.
 batch_normalization_1_4374093795:.
 batch_normalization_1_4374093797:.
 batch_normalization_1_4374093799:.
 batch_normalization_1_4374093801:
identity��-batch_normalization_1/StatefulPartitionedCall� conv2d_1/StatefulPartitionedCall�
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_1_4374093790conv2d_1_4374093792*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_conv2d_1_layer_call_and_return_conditional_losses_4374093714�
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0 batch_normalization_1_4374093795 batch_normalization_1_4374093797 batch_normalization_1_4374093799 batch_normalization_1_4374093801*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *^
fYRW
U__inference_batch_normalization_1_layer_call_and_return_conditional_losses_4374093686�
re_lu_1/PartitionedCallPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_re_lu_1_layer_call_and_return_conditional_losses_4374093734w
IdentityIdentity re_lu_1/PartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:����������
NoOpNoOp.^batch_normalization_1/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':���������	: : : : : : 2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall:W S
/
_output_shapes
:���������	
 
_user_specified_nameinputs
�^
�
C__inference_dqn_layer_call_and_return_conditional_losses_4374094449
xJ
0sequential_conv2d_conv2d_readvariableop_resource:?
1sequential_conv2d_biasadd_readvariableop_resource:D
6sequential_batch_normalization_readvariableop_resource:F
8sequential_batch_normalization_readvariableop_1_resource:U
Gsequential_batch_normalization_fusedbatchnormv3_readvariableop_resource:W
Isequential_batch_normalization_fusedbatchnormv3_readvariableop_1_resource:N
4sequential_1_conv2d_1_conv2d_readvariableop_resource:C
5sequential_1_conv2d_1_biasadd_readvariableop_resource:H
:sequential_1_batch_normalization_1_readvariableop_resource:J
<sequential_1_batch_normalization_1_readvariableop_1_resource:Y
Ksequential_1_batch_normalization_1_fusedbatchnormv3_readvariableop_resource:[
Msequential_1_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource:6
$dense_matmul_readvariableop_resource:@3
%dense_biasadd_readvariableop_resource:6
&dense_1_matmul_readvariableop_resource:
 	5
'dense_1_biasadd_readvariableop_resource:	
identity��dense/BiasAdd/ReadVariableOp�dense/MatMul/ReadVariableOp�dense_1/BiasAdd/ReadVariableOp�dense_1/MatMul/ReadVariableOp�>sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp�@sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1�-sequential/batch_normalization/ReadVariableOp�/sequential/batch_normalization/ReadVariableOp_1�(sequential/conv2d/BiasAdd/ReadVariableOp�'sequential/conv2d/Conv2D/ReadVariableOp�Bsequential_1/batch_normalization_1/FusedBatchNormV3/ReadVariableOp�Dsequential_1/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1�1sequential_1/batch_normalization_1/ReadVariableOp�3sequential_1/batch_normalization_1/ReadVariableOp_1�,sequential_1/conv2d_1/BiasAdd/ReadVariableOp�+sequential_1/conv2d_1/Conv2D/ReadVariableOp�
'sequential/conv2d/Conv2D/ReadVariableOpReadVariableOp0sequential_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
sequential/conv2d/Conv2DConv2Dx/sequential/conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������	*
paddingVALID*
strides
�
(sequential/conv2d/BiasAdd/ReadVariableOpReadVariableOp1sequential_conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
sequential/conv2d/BiasAddBiasAdd!sequential/conv2d/Conv2D:output:00sequential/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������	�
-sequential/batch_normalization/ReadVariableOpReadVariableOp6sequential_batch_normalization_readvariableop_resource*
_output_shapes
:*
dtype0�
/sequential/batch_normalization/ReadVariableOp_1ReadVariableOp8sequential_batch_normalization_readvariableop_1_resource*
_output_shapes
:*
dtype0�
>sequential/batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOpGsequential_batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0�
@sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpIsequential_batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0�
/sequential/batch_normalization/FusedBatchNormV3FusedBatchNormV3"sequential/conv2d/BiasAdd:output:05sequential/batch_normalization/ReadVariableOp:value:07sequential/batch_normalization/ReadVariableOp_1:value:0Fsequential/batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0Hsequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������	:::::*
epsilon%o�:*
is_training( �
sequential/re_lu/ReluRelu3sequential/batch_normalization/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:���������	�
+sequential_1/conv2d_1/Conv2D/ReadVariableOpReadVariableOp4sequential_1_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
sequential_1/conv2d_1/Conv2DConv2D#sequential/re_lu/Relu:activations:03sequential_1/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
�
,sequential_1/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp5sequential_1_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
sequential_1/conv2d_1/BiasAddBiasAdd%sequential_1/conv2d_1/Conv2D:output:04sequential_1/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:����������
1sequential_1/batch_normalization_1/ReadVariableOpReadVariableOp:sequential_1_batch_normalization_1_readvariableop_resource*
_output_shapes
:*
dtype0�
3sequential_1/batch_normalization_1/ReadVariableOp_1ReadVariableOp<sequential_1_batch_normalization_1_readvariableop_1_resource*
_output_shapes
:*
dtype0�
Bsequential_1/batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOpKsequential_1_batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0�
Dsequential_1/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpMsequential_1_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0�
3sequential_1/batch_normalization_1/FusedBatchNormV3FusedBatchNormV3&sequential_1/conv2d_1/BiasAdd:output:09sequential_1/batch_normalization_1/ReadVariableOp:value:0;sequential_1/batch_normalization_1/ReadVariableOp_1:value:0Jsequential_1/batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0Lsequential_1/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������:::::*
epsilon%o�:*
is_training( �
sequential_1/re_lu_1/ReluRelu7sequential_1/batch_normalization_1/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:���������^
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"����@   �
flatten/ReshapeReshape'sequential_1/re_lu_1/Relu:activations:0flatten/Const:output:0*
T0*'
_output_shapes
:���������@d
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        f
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    `   f
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
strided_sliceStridedSliceflatten/Reshape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������@*

begin_mask*
end_maskf
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    `   h
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        h
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
strided_slice_1StridedSliceflatten/Reshape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*%
_output_shapes
:��������� *

begin_mask*
end_mask�
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0�
dense/MatMulMatMulstrided_slice:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

: 	*
dtype0�
dense_1/MatMulMatMulstrided_slice_1:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������	�
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:	*
dtype0�
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������	X
Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :u
MeanMeandense_1/BiasAdd:output:0Mean/reduction_indices:output:0*
T0*#
_output_shapes
:���������^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"����   k
ReshapeReshapeMean:output:0Reshape/shape:output:0*
T0*'
_output_shapes
:���������h
subSubdense_1/BiasAdd:output:0Reshape:output:0*
T0*'
_output_shapes
:���������	_
addAddV2dense/BiasAdd:output:0sub:z:0*
T0*'
_output_shapes
:���������	V
IdentityIdentityadd:z:0^NoOp*
T0*'
_output_shapes
:���������	�
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp?^sequential/batch_normalization/FusedBatchNormV3/ReadVariableOpA^sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1.^sequential/batch_normalization/ReadVariableOp0^sequential/batch_normalization/ReadVariableOp_1)^sequential/conv2d/BiasAdd/ReadVariableOp(^sequential/conv2d/Conv2D/ReadVariableOpC^sequential_1/batch_normalization_1/FusedBatchNormV3/ReadVariableOpE^sequential_1/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_12^sequential_1/batch_normalization_1/ReadVariableOp4^sequential_1/batch_normalization_1/ReadVariableOp_1-^sequential_1/conv2d_1/BiasAdd/ReadVariableOp,^sequential_1/conv2d_1/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:���������: : : : : : : : : : : : : : : : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2�
>sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp>sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp2�
@sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1@sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_12^
-sequential/batch_normalization/ReadVariableOp-sequential/batch_normalization/ReadVariableOp2b
/sequential/batch_normalization/ReadVariableOp_1/sequential/batch_normalization/ReadVariableOp_12T
(sequential/conv2d/BiasAdd/ReadVariableOp(sequential/conv2d/BiasAdd/ReadVariableOp2R
'sequential/conv2d/Conv2D/ReadVariableOp'sequential/conv2d/Conv2D/ReadVariableOp2�
Bsequential_1/batch_normalization_1/FusedBatchNormV3/ReadVariableOpBsequential_1/batch_normalization_1/FusedBatchNormV3/ReadVariableOp2�
Dsequential_1/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1Dsequential_1/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_12f
1sequential_1/batch_normalization_1/ReadVariableOp1sequential_1/batch_normalization_1/ReadVariableOp2j
3sequential_1/batch_normalization_1/ReadVariableOp_13sequential_1/batch_normalization_1/ReadVariableOp_12\
,sequential_1/conv2d_1/BiasAdd/ReadVariableOp,sequential_1/conv2d_1/BiasAdd/ReadVariableOp2Z
+sequential_1/conv2d_1/Conv2D/ReadVariableOp+sequential_1/conv2d_1/Conv2D/ReadVariableOp:R N
/
_output_shapes
:���������

_user_specified_namex
�	
�
/__inference_sequential_layer_call_fn_4374093509
conv2d_input!
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallconv2d_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������	*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *S
fNRL
J__inference_sequential_layer_call_and_return_conditional_losses_4374093494w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������	`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':���������: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
/
_output_shapes
:���������
&
_user_specified_nameconv2d_input
�
�
U__inference_batch_normalization_1_layer_call_and_return_conditional_losses_4374093686

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������:::::*
epsilon%o�:*
exponential_avg_factor%
�#<�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+����������������������������
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+���������������������������: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
�,
�
C__inference_dqn_layer_call_and_return_conditional_losses_4374094246
input_1/
sequential_4374094194:#
sequential_4374094196:#
sequential_4374094198:#
sequential_4374094200:#
sequential_4374094202:#
sequential_4374094204:1
sequential_1_4374094207:%
sequential_1_4374094209:%
sequential_1_4374094211:%
sequential_1_4374094213:%
sequential_1_4374094215:%
sequential_1_4374094217:"
dense_4374094229:@
dense_4374094231:"
dense_1_4374094234:
 	 
dense_1_4374094236:	
identity��dense/StatefulPartitionedCall�dense_1/StatefulPartitionedCall�"sequential/StatefulPartitionedCall�$sequential_1/StatefulPartitionedCall�
"sequential/StatefulPartitionedCallStatefulPartitionedCallinput_1sequential_4374094194sequential_4374094196sequential_4374094198sequential_4374094200sequential_4374094202sequential_4374094204*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������	*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *S
fNRL
J__inference_sequential_layer_call_and_return_conditional_losses_4374093494�
$sequential_1/StatefulPartitionedCallStatefulPartitionedCall+sequential/StatefulPartitionedCall:output:0sequential_1_4374094207sequential_1_4374094209sequential_1_4374094211sequential_1_4374094213sequential_1_4374094215sequential_1_4374094217*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *U
fPRN
L__inference_sequential_1_layer_call_and_return_conditional_losses_4374093737�
flatten/PartitionedCallPartitionedCall-sequential_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_flatten_layer_call_and_return_conditional_losses_4374093915d
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        f
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    `   f
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
strided_sliceStridedSlice flatten/PartitionedCall:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������@*

begin_mask*
end_maskf
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    `   h
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        h
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
strided_slice_1StridedSlice flatten/PartitionedCall:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*%
_output_shapes
:��������� *

begin_mask*
end_mask�
dense/StatefulPartitionedCallStatefulPartitionedCallstrided_slice:output:0dense_4374094229dense_4374094231*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dense_layer_call_and_return_conditional_losses_4374093935�
dense_1/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_1:output:0dense_1_4374094234dense_1_4374094236*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������	*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_dense_1_layer_call_and_return_conditional_losses_4374093951X
Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :�
MeanMean(dense_1/StatefulPartitionedCall:output:0Mean/reduction_indices:output:0*
T0*#
_output_shapes
:���������^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"����   k
ReshapeReshapeMean:output:0Reshape/shape:output:0*
T0*'
_output_shapes
:���������x
subSub(dense_1/StatefulPartitionedCall:output:0Reshape:output:0*
T0*'
_output_shapes
:���������	o
addAddV2&dense/StatefulPartitionedCall:output:0sub:z:0*
T0*'
_output_shapes
:���������	V
IdentityIdentityadd:z:0^NoOp*
T0*'
_output_shapes
:���������	�
NoOpNoOp^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall#^sequential/StatefulPartitionedCall%^sequential_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:���������: : : : : : : : : : : : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2H
"sequential/StatefulPartitionedCall"sequential/StatefulPartitionedCall2L
$sequential_1/StatefulPartitionedCall$sequential_1/StatefulPartitionedCall:X T
/
_output_shapes
:���������
!
_user_specified_name	input_1
�
�
U__inference_batch_normalization_1_layer_call_and_return_conditional_losses_4374093655

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������:::::*
epsilon%o�:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+����������������������������
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+���������������������������: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
�A
�

&__inference__traced_restore_4374095090
file_prefix3
!assignvariableop_dqn_dense_kernel:@/
!assignvariableop_1_dqn_dense_bias:5
%assignvariableop_2_dqn_dense_1_kernel:
 	1
#assignvariableop_3_dqn_dense_1_bias:	:
 assignvariableop_4_conv2d_kernel:,
assignvariableop_5_conv2d_bias::
,assignvariableop_6_batch_normalization_gamma:9
+assignvariableop_7_batch_normalization_beta:@
2assignvariableop_8_batch_normalization_moving_mean:D
6assignvariableop_9_batch_normalization_moving_variance:=
#assignvariableop_10_conv2d_1_kernel:/
!assignvariableop_11_conv2d_1_bias:=
/assignvariableop_12_batch_normalization_1_gamma:<
.assignvariableop_13_batch_normalization_1_beta:C
5assignvariableop_14_batch_normalization_1_moving_mean:G
9assignvariableop_15_batch_normalization_1_moving_variance:
identity_17��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_2�AssignVariableOp_3�AssignVariableOp_4�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B-state_value/kernel/.ATTRIBUTES/VARIABLE_VALUEB+state_value/bias/.ATTRIBUTES/VARIABLE_VALUEB+advantage/kernel/.ATTRIBUTES/VARIABLE_VALUEB)advantage/bias/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*5
value,B*B B B B B B B B B B B B B B B B B �
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*X
_output_shapesF
D:::::::::::::::::*
dtypes
2[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOpAssignVariableOp!assignvariableop_dqn_dense_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_1AssignVariableOp!assignvariableop_1_dqn_dense_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_2AssignVariableOp%assignvariableop_2_dqn_dense_1_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_3AssignVariableOp#assignvariableop_3_dqn_dense_1_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_4AssignVariableOp assignvariableop_4_conv2d_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_5AssignVariableOpassignvariableop_5_conv2d_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_6AssignVariableOp,assignvariableop_6_batch_normalization_gammaIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_7AssignVariableOp+assignvariableop_7_batch_normalization_betaIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_8AssignVariableOp2assignvariableop_8_batch_normalization_moving_meanIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_9AssignVariableOp6assignvariableop_9_batch_normalization_moving_varianceIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_10AssignVariableOp#assignvariableop_10_conv2d_1_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_11AssignVariableOp!assignvariableop_11_conv2d_1_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_12AssignVariableOp/assignvariableop_12_batch_normalization_1_gammaIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_13AssignVariableOp.assignvariableop_13_batch_normalization_1_betaIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_14AssignVariableOp5assignvariableop_14_batch_normalization_1_moving_meanIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_15AssignVariableOp9assignvariableop_15_batch_normalization_1_moving_varianceIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 �
Identity_16Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_17IdentityIdentity_16:output:0^NoOp_1*
T0*
_output_shapes
: �
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_17Identity_17:output:0*5
_input_shapes$
": : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�$
�
L__inference_sequential_1_layer_call_and_return_conditional_losses_4374094730

inputsA
'conv2d_1_conv2d_readvariableop_resource:6
(conv2d_1_biasadd_readvariableop_resource:;
-batch_normalization_1_readvariableop_resource:=
/batch_normalization_1_readvariableop_1_resource:L
>batch_normalization_1_fusedbatchnormv3_readvariableop_resource:N
@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource:
identity��$batch_normalization_1/AssignNewValue�&batch_normalization_1/AssignNewValue_1�5batch_normalization_1/FusedBatchNormV3/ReadVariableOp�7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1�$batch_normalization_1/ReadVariableOp�&batch_normalization_1/ReadVariableOp_1�conv2d_1/BiasAdd/ReadVariableOp�conv2d_1/Conv2D/ReadVariableOp�
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
conv2d_1/Conv2DConv2Dinputs&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
�
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:����������
$batch_normalization_1/ReadVariableOpReadVariableOp-batch_normalization_1_readvariableop_resource*
_output_shapes
:*
dtype0�
&batch_normalization_1/ReadVariableOp_1ReadVariableOp/batch_normalization_1_readvariableop_1_resource*
_output_shapes
:*
dtype0�
5batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0�
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0�
&batch_normalization_1/FusedBatchNormV3FusedBatchNormV3conv2d_1/BiasAdd:output:0,batch_normalization_1/ReadVariableOp:value:0.batch_normalization_1/ReadVariableOp_1:value:0=batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������:::::*
epsilon%o�:*
exponential_avg_factor%
�#<�
$batch_normalization_1/AssignNewValueAssignVariableOp>batch_normalization_1_fusedbatchnormv3_readvariableop_resource3batch_normalization_1/FusedBatchNormV3:batch_mean:06^batch_normalization_1/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0�
&batch_normalization_1/AssignNewValue_1AssignVariableOp@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_1/FusedBatchNormV3:batch_variance:08^batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0z
re_lu_1/ReluRelu*batch_normalization_1/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:���������q
IdentityIdentityre_lu_1/Relu:activations:0^NoOp*
T0*/
_output_shapes
:����������
NoOpNoOp%^batch_normalization_1/AssignNewValue'^batch_normalization_1/AssignNewValue_16^batch_normalization_1/FusedBatchNormV3/ReadVariableOp8^batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_1/ReadVariableOp'^batch_normalization_1/ReadVariableOp_1 ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':���������	: : : : : : 2L
$batch_normalization_1/AssignNewValue$batch_normalization_1/AssignNewValue2P
&batch_normalization_1/AssignNewValue_1&batch_normalization_1/AssignNewValue_12n
5batch_normalization_1/FusedBatchNormV3/ReadVariableOp5batch_normalization_1/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_17batch_normalization_1/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_1/ReadVariableOp$batch_normalization_1/ReadVariableOp2P
&batch_normalization_1/ReadVariableOp_1&batch_normalization_1/ReadVariableOp_12B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp:W S
/
_output_shapes
:���������	
 
_user_specified_nameinputs
�	
�
/__inference_sequential_layer_call_fn_4374094596

inputs!
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������	*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *S
fNRL
J__inference_sequential_layer_call_and_return_conditional_losses_4374093563w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������	`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':���������: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�

�
H__inference_conv2d_1_layer_call_and_return_conditional_losses_4374094889

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������	: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������	
 
_user_specified_nameinputs
�,
�
C__inference_dqn_layer_call_and_return_conditional_losses_4374094301
input_1/
sequential_4374094249:#
sequential_4374094251:#
sequential_4374094253:#
sequential_4374094255:#
sequential_4374094257:#
sequential_4374094259:1
sequential_1_4374094262:%
sequential_1_4374094264:%
sequential_1_4374094266:%
sequential_1_4374094268:%
sequential_1_4374094270:%
sequential_1_4374094272:"
dense_4374094284:@
dense_4374094286:"
dense_1_4374094289:
 	 
dense_1_4374094291:	
identity��dense/StatefulPartitionedCall�dense_1/StatefulPartitionedCall�"sequential/StatefulPartitionedCall�$sequential_1/StatefulPartitionedCall�
"sequential/StatefulPartitionedCallStatefulPartitionedCallinput_1sequential_4374094249sequential_4374094251sequential_4374094253sequential_4374094255sequential_4374094257sequential_4374094259*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������	*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *S
fNRL
J__inference_sequential_layer_call_and_return_conditional_losses_4374093563�
$sequential_1/StatefulPartitionedCallStatefulPartitionedCall+sequential/StatefulPartitionedCall:output:0sequential_1_4374094262sequential_1_4374094264sequential_1_4374094266sequential_1_4374094268sequential_1_4374094270sequential_1_4374094272*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *U
fPRN
L__inference_sequential_1_layer_call_and_return_conditional_losses_4374093806�
flatten/PartitionedCallPartitionedCall-sequential_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_flatten_layer_call_and_return_conditional_losses_4374093915d
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        f
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    `   f
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
strided_sliceStridedSlice flatten/PartitionedCall:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������@*

begin_mask*
end_maskf
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    `   h
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        h
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
strided_slice_1StridedSlice flatten/PartitionedCall:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*%
_output_shapes
:��������� *

begin_mask*
end_mask�
dense/StatefulPartitionedCallStatefulPartitionedCallstrided_slice:output:0dense_4374094284dense_4374094286*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dense_layer_call_and_return_conditional_losses_4374093935�
dense_1/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_1:output:0dense_1_4374094289dense_1_4374094291*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������	*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_dense_1_layer_call_and_return_conditional_losses_4374093951X
Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :�
MeanMean(dense_1/StatefulPartitionedCall:output:0Mean/reduction_indices:output:0*
T0*#
_output_shapes
:���������^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"����   k
ReshapeReshapeMean:output:0Reshape/shape:output:0*
T0*'
_output_shapes
:���������x
subSub(dense_1/StatefulPartitionedCall:output:0Reshape:output:0*
T0*'
_output_shapes
:���������	o
addAddV2&dense/StatefulPartitionedCall:output:0sub:z:0*
T0*'
_output_shapes
:���������	V
IdentityIdentityadd:z:0^NoOp*
T0*'
_output_shapes
:���������	�
NoOpNoOp^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall#^sequential/StatefulPartitionedCall%^sequential_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:���������: : : : : : : : : : : : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2H
"sequential/StatefulPartitionedCall"sequential/StatefulPartitionedCall2L
$sequential_1/StatefulPartitionedCall$sequential_1/StatefulPartitionedCall:X T
/
_output_shapes
:���������
!
_user_specified_name	input_1
�

�
F__inference_conv2d_layer_call_and_return_conditional_losses_4374094798

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������	*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������	g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:���������	w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
J__inference_sequential_layer_call_and_return_conditional_losses_4374093563

inputs+
conv2d_4374093547:
conv2d_4374093549:,
batch_normalization_4374093552:,
batch_normalization_4374093554:,
batch_normalization_4374093556:,
batch_normalization_4374093558:
identity��+batch_normalization/StatefulPartitionedCall�conv2d/StatefulPartitionedCall�
conv2d/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_4374093547conv2d_4374093549*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������	*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_conv2d_layer_call_and_return_conditional_losses_4374093471�
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0batch_normalization_4374093552batch_normalization_4374093554batch_normalization_4374093556batch_normalization_4374093558*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������	*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *\
fWRU
S__inference_batch_normalization_layer_call_and_return_conditional_losses_4374093443�
re_lu/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������	* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_re_lu_layer_call_and_return_conditional_losses_4374093491u
IdentityIdentityre_lu/PartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������	�
NoOpNoOp,^batch_normalization/StatefulPartitionedCall^conv2d/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':���������: : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
c
G__inference_re_lu_1_layer_call_and_return_conditional_losses_4374093734

inputs
identityN
ReluReluinputs*
T0*/
_output_shapes
:���������b
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�

�
F__inference_conv2d_layer_call_and_return_conditional_losses_4374093471

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������	*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������	g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:���������	w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�	
�
G__inference_dense_1_layer_call_and_return_conditional_losses_4374094779

inputs.
matmul_readvariableop_resource:
 	-
biasadd_readvariableop_resource:	
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpr
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: 	*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������	r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:	*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������	_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������	w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:��������� : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:M I
%
_output_shapes
:��������� 
 
_user_specified_nameinputs
�,
�
C__inference_dqn_layer_call_and_return_conditional_losses_4374094119
x/
sequential_4374094067:#
sequential_4374094069:#
sequential_4374094071:#
sequential_4374094073:#
sequential_4374094075:#
sequential_4374094077:1
sequential_1_4374094080:%
sequential_1_4374094082:%
sequential_1_4374094084:%
sequential_1_4374094086:%
sequential_1_4374094088:%
sequential_1_4374094090:"
dense_4374094102:@
dense_4374094104:"
dense_1_4374094107:
 	 
dense_1_4374094109:	
identity��dense/StatefulPartitionedCall�dense_1/StatefulPartitionedCall�"sequential/StatefulPartitionedCall�$sequential_1/StatefulPartitionedCall�
"sequential/StatefulPartitionedCallStatefulPartitionedCallxsequential_4374094067sequential_4374094069sequential_4374094071sequential_4374094073sequential_4374094075sequential_4374094077*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������	*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *S
fNRL
J__inference_sequential_layer_call_and_return_conditional_losses_4374093563�
$sequential_1/StatefulPartitionedCallStatefulPartitionedCall+sequential/StatefulPartitionedCall:output:0sequential_1_4374094080sequential_1_4374094082sequential_1_4374094084sequential_1_4374094086sequential_1_4374094088sequential_1_4374094090*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *U
fPRN
L__inference_sequential_1_layer_call_and_return_conditional_losses_4374093806�
flatten/PartitionedCallPartitionedCall-sequential_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_flatten_layer_call_and_return_conditional_losses_4374093915d
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        f
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    `   f
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
strided_sliceStridedSlice flatten/PartitionedCall:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������@*

begin_mask*
end_maskf
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    `   h
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        h
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      �
strided_slice_1StridedSlice flatten/PartitionedCall:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*%
_output_shapes
:��������� *

begin_mask*
end_mask�
dense/StatefulPartitionedCallStatefulPartitionedCallstrided_slice:output:0dense_4374094102dense_4374094104*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_dense_layer_call_and_return_conditional_losses_4374093935�
dense_1/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_1:output:0dense_1_4374094107dense_1_4374094109*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������	*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_dense_1_layer_call_and_return_conditional_losses_4374093951X
Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :�
MeanMean(dense_1/StatefulPartitionedCall:output:0Mean/reduction_indices:output:0*
T0*#
_output_shapes
:���������^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"����   k
ReshapeReshapeMean:output:0Reshape/shape:output:0*
T0*'
_output_shapes
:���������x
subSub(dense_1/StatefulPartitionedCall:output:0Reshape:output:0*
T0*'
_output_shapes
:���������	o
addAddV2&dense/StatefulPartitionedCall:output:0sub:z:0*
T0*'
_output_shapes
:���������	V
IdentityIdentityadd:z:0^NoOp*
T0*'
_output_shapes
:���������	�
NoOpNoOp^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall#^sequential/StatefulPartitionedCall%^sequential_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:���������: : : : : : : : : : : : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2H
"sequential/StatefulPartitionedCall"sequential/StatefulPartitionedCall2L
$sequential_1/StatefulPartitionedCall$sequential_1/StatefulPartitionedCall:R N
/
_output_shapes
:���������

_user_specified_namex
�
�
S__inference_batch_normalization_layer_call_and_return_conditional_losses_4374093443

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������:::::*
epsilon%o�:*
exponential_avg_factor%
�#<�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+����������������������������
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+���������������������������: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
�
�
-__inference_conv2d_1_layer_call_fn_4374094879

inputs!
unknown:
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Q
fLRJ
H__inference_conv2d_1_layer_call_and_return_conditional_losses_4374093714w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������	: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������	
 
_user_specified_nameinputs
�
�
U__inference_batch_normalization_1_layer_call_and_return_conditional_losses_4374094933

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������:::::*
epsilon%o�:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+����������������������������
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+���������������������������: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs"�L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
C
input_18
serving_default_input_1:0���������<
output_10
StatefulPartitionedCall:0���������	tensorflow/serving/predict:��
�
	conv1
	conv2
flatten
state_value
	advantage
	variables
trainable_variables
regularization_losses
		keras_api

__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures"
_tf_keras_model
�
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer-2
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_sequential
�
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer-2
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_sequential
�
 	variables
!trainable_variables
"regularization_losses
#	keras_api
$__call__
*%&call_and_return_all_conditional_losses"
_tf_keras_layer
�

&kernel
'bias
(	variables
)trainable_variables
*regularization_losses
+	keras_api
,__call__
*-&call_and_return_all_conditional_losses"
_tf_keras_layer
�

.kernel
/bias
0	variables
1trainable_variables
2regularization_losses
3	keras_api
4__call__
*5&call_and_return_all_conditional_losses"
_tf_keras_layer
�
60
71
82
93
:4
;5
<6
=7
>8
?9
@10
A11
&12
'13
.14
/15"
trackable_list_wrapper
v
60
71
82
93
<4
=5
>6
?7
&8
'9
.10
/11"
trackable_list_wrapper
 "
trackable_list_wrapper
�
Bnon_trainable_variables

Clayers
Dmetrics
Elayer_regularization_losses
Flayer_metrics
	variables
trainable_variables
regularization_losses

__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�2�
(__inference_dqn_layer_call_fn_4374093999
(__inference_dqn_layer_call_fn_4374094338
(__inference_dqn_layer_call_fn_4374094375
(__inference_dqn_layer_call_fn_4374094191�
���
FullArgSpec$
args�
jself
jx

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
C__inference_dqn_layer_call_and_return_conditional_losses_4374094449
C__inference_dqn_layer_call_and_return_conditional_losses_4374094523
C__inference_dqn_layer_call_and_return_conditional_losses_4374094246
C__inference_dqn_layer_call_and_return_conditional_losses_4374094301�
���
FullArgSpec$
args�
jself
jx

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
%__inference__wrapped_model_4374093390input_1"�
���
FullArgSpec
args� 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
,
Gserving_default"
signature_map
�

6kernel
7bias
H	variables
Itrainable_variables
Jregularization_losses
K	keras_api
L__call__
*M&call_and_return_all_conditional_losses"
_tf_keras_layer
�
Naxis
	8gamma
9beta
:moving_mean
;moving_variance
O	variables
Ptrainable_variables
Qregularization_losses
R	keras_api
S__call__
*T&call_and_return_all_conditional_losses"
_tf_keras_layer
�
U	variables
Vtrainable_variables
Wregularization_losses
X	keras_api
Y__call__
*Z&call_and_return_all_conditional_losses"
_tf_keras_layer
J
60
71
82
93
:4
;5"
trackable_list_wrapper
<
60
71
82
93"
trackable_list_wrapper
 "
trackable_list_wrapper
�
[non_trainable_variables

\layers
]metrics
^layer_regularization_losses
_layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�2�
/__inference_sequential_layer_call_fn_4374093509
/__inference_sequential_layer_call_fn_4374094579
/__inference_sequential_layer_call_fn_4374094596
/__inference_sequential_layer_call_fn_4374093595�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
J__inference_sequential_layer_call_and_return_conditional_losses_4374094621
J__inference_sequential_layer_call_and_return_conditional_losses_4374094646
J__inference_sequential_layer_call_and_return_conditional_losses_4374093614
J__inference_sequential_layer_call_and_return_conditional_losses_4374093633�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�

<kernel
=bias
`	variables
atrainable_variables
bregularization_losses
c	keras_api
d__call__
*e&call_and_return_all_conditional_losses"
_tf_keras_layer
�
faxis
	>gamma
?beta
@moving_mean
Amoving_variance
g	variables
htrainable_variables
iregularization_losses
j	keras_api
k__call__
*l&call_and_return_all_conditional_losses"
_tf_keras_layer
�
m	variables
ntrainable_variables
oregularization_losses
p	keras_api
q__call__
*r&call_and_return_all_conditional_losses"
_tf_keras_layer
J
<0
=1
>2
?3
@4
A5"
trackable_list_wrapper
<
<0
=1
>2
?3"
trackable_list_wrapper
 "
trackable_list_wrapper
�
snon_trainable_variables

tlayers
umetrics
vlayer_regularization_losses
wlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�2�
1__inference_sequential_1_layer_call_fn_4374093752
1__inference_sequential_1_layer_call_fn_4374094663
1__inference_sequential_1_layer_call_fn_4374094680
1__inference_sequential_1_layer_call_fn_4374093838�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
L__inference_sequential_1_layer_call_and_return_conditional_losses_4374094705
L__inference_sequential_1_layer_call_and_return_conditional_losses_4374094730
L__inference_sequential_1_layer_call_and_return_conditional_losses_4374093857
L__inference_sequential_1_layer_call_and_return_conditional_losses_4374093876�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
xnon_trainable_variables

ylayers
zmetrics
{layer_regularization_losses
|layer_metrics
 	variables
!trainable_variables
"regularization_losses
$__call__
*%&call_and_return_all_conditional_losses
&%"call_and_return_conditional_losses"
_generic_user_object
�2�
,__inference_flatten_layer_call_fn_4374094735�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
G__inference_flatten_layer_call_and_return_conditional_losses_4374094741�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
": @2dqn/dense/kernel
:2dqn/dense/bias
.
&0
'1"
trackable_list_wrapper
.
&0
'1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
}non_trainable_variables

~layers
metrics
 �layer_regularization_losses
�layer_metrics
(	variables
)trainable_variables
*regularization_losses
,__call__
*-&call_and_return_all_conditional_losses
&-"call_and_return_conditional_losses"
_generic_user_object
�2�
*__inference_dense_layer_call_fn_4374094750�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
E__inference_dense_layer_call_and_return_conditional_losses_4374094760�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
":  	2dqn/dense_1/kernel
:	2dqn/dense_1/bias
.
.0
/1"
trackable_list_wrapper
.
.0
/1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
0	variables
1trainable_variables
2regularization_losses
4__call__
*5&call_and_return_all_conditional_losses
&5"call_and_return_conditional_losses"
_generic_user_object
�2�
,__inference_dense_1_layer_call_fn_4374094769�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
G__inference_dense_1_layer_call_and_return_conditional_losses_4374094779�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
':%2conv2d/kernel
:2conv2d/bias
':%2batch_normalization/gamma
&:$2batch_normalization/beta
/:- (2batch_normalization/moving_mean
3:1 (2#batch_normalization/moving_variance
):'2conv2d_1/kernel
:2conv2d_1/bias
):'2batch_normalization_1/gamma
(:&2batch_normalization_1/beta
1:/ (2!batch_normalization_1/moving_mean
5:3 (2%batch_normalization_1/moving_variance
<
:0
;1
@2
A3"
trackable_list_wrapper
C
0
1
2
3
4"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
(__inference_signature_wrapper_4374094562input_1"�
���
FullArgSpec
args� 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
.
60
71"
trackable_list_wrapper
.
60
71"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
H	variables
Itrainable_variables
Jregularization_losses
L__call__
*M&call_and_return_all_conditional_losses
&M"call_and_return_conditional_losses"
_generic_user_object
�2�
+__inference_conv2d_layer_call_fn_4374094788�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
F__inference_conv2d_layer_call_and_return_conditional_losses_4374094798�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
<
80
91
:2
;3"
trackable_list_wrapper
.
80
91"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
O	variables
Ptrainable_variables
Qregularization_losses
S__call__
*T&call_and_return_all_conditional_losses
&T"call_and_return_conditional_losses"
_generic_user_object
�2�
8__inference_batch_normalization_layer_call_fn_4374094811
8__inference_batch_normalization_layer_call_fn_4374094824�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
S__inference_batch_normalization_layer_call_and_return_conditional_losses_4374094842
S__inference_batch_normalization_layer_call_and_return_conditional_losses_4374094860�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
U	variables
Vtrainable_variables
Wregularization_losses
Y__call__
*Z&call_and_return_all_conditional_losses
&Z"call_and_return_conditional_losses"
_generic_user_object
�2�
*__inference_re_lu_layer_call_fn_4374094865�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
E__inference_re_lu_layer_call_and_return_conditional_losses_4374094870�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
.
:0
;1"
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
<0
=1"
trackable_list_wrapper
.
<0
=1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
`	variables
atrainable_variables
bregularization_losses
d__call__
*e&call_and_return_all_conditional_losses
&e"call_and_return_conditional_losses"
_generic_user_object
�2�
-__inference_conv2d_1_layer_call_fn_4374094879�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
H__inference_conv2d_1_layer_call_and_return_conditional_losses_4374094889�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
<
>0
?1
@2
A3"
trackable_list_wrapper
.
>0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
g	variables
htrainable_variables
iregularization_losses
k__call__
*l&call_and_return_all_conditional_losses
&l"call_and_return_conditional_losses"
_generic_user_object
�2�
:__inference_batch_normalization_1_layer_call_fn_4374094902
:__inference_batch_normalization_1_layer_call_fn_4374094915�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
U__inference_batch_normalization_1_layer_call_and_return_conditional_losses_4374094933
U__inference_batch_normalization_1_layer_call_and_return_conditional_losses_4374094951�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
m	variables
ntrainable_variables
oregularization_losses
q__call__
*r&call_and_return_all_conditional_losses
&r"call_and_return_conditional_losses"
_generic_user_object
�2�
,__inference_re_lu_1_layer_call_fn_4374094956�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
G__inference_re_lu_1_layer_call_and_return_conditional_losses_4374094961�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
.
@0
A1"
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
:0
;1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
@0
A1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper�
%__inference__wrapped_model_4374093390�6789:;<=>?@A&'./8�5
.�+
)�&
input_1���������
� "3�0
.
output_1"�
output_1���������	�
U__inference_batch_normalization_1_layer_call_and_return_conditional_losses_4374094933�>?@AM�J
C�@
:�7
inputs+���������������������������
p 
� "?�<
5�2
0+���������������������������
� �
U__inference_batch_normalization_1_layer_call_and_return_conditional_losses_4374094951�>?@AM�J
C�@
:�7
inputs+���������������������������
p
� "?�<
5�2
0+���������������������������
� �
:__inference_batch_normalization_1_layer_call_fn_4374094902�>?@AM�J
C�@
:�7
inputs+���������������������������
p 
� "2�/+����������������������������
:__inference_batch_normalization_1_layer_call_fn_4374094915�>?@AM�J
C�@
:�7
inputs+���������������������������
p
� "2�/+����������������������������
S__inference_batch_normalization_layer_call_and_return_conditional_losses_4374094842�89:;M�J
C�@
:�7
inputs+���������������������������
p 
� "?�<
5�2
0+���������������������������
� �
S__inference_batch_normalization_layer_call_and_return_conditional_losses_4374094860�89:;M�J
C�@
:�7
inputs+���������������������������
p
� "?�<
5�2
0+���������������������������
� �
8__inference_batch_normalization_layer_call_fn_4374094811�89:;M�J
C�@
:�7
inputs+���������������������������
p 
� "2�/+����������������������������
8__inference_batch_normalization_layer_call_fn_4374094824�89:;M�J
C�@
:�7
inputs+���������������������������
p
� "2�/+����������������������������
H__inference_conv2d_1_layer_call_and_return_conditional_losses_4374094889l<=7�4
-�*
(�%
inputs���������	
� "-�*
#� 
0���������
� �
-__inference_conv2d_1_layer_call_fn_4374094879_<=7�4
-�*
(�%
inputs���������	
� " �����������
F__inference_conv2d_layer_call_and_return_conditional_losses_4374094798l677�4
-�*
(�%
inputs���������
� "-�*
#� 
0���������	
� �
+__inference_conv2d_layer_call_fn_4374094788_677�4
-�*
(�%
inputs���������
� " ����������	�
G__inference_dense_1_layer_call_and_return_conditional_losses_4374094779Z./-�*
#� 
�
inputs��������� 
� "%�"
�
0���������	
� }
,__inference_dense_1_layer_call_fn_4374094769M./-�*
#� 
�
inputs��������� 
� "����������	�
E__inference_dense_layer_call_and_return_conditional_losses_4374094760\&'/�,
%�"
 �
inputs���������@
� "%�"
�
0���������
� }
*__inference_dense_layer_call_fn_4374094750O&'/�,
%�"
 �
inputs���������@
� "�����������
C__inference_dqn_layer_call_and_return_conditional_losses_4374094246w6789:;<=>?@A&'./<�9
2�/
)�&
input_1���������
p 
� "%�"
�
0���������	
� �
C__inference_dqn_layer_call_and_return_conditional_losses_4374094301w6789:;<=>?@A&'./<�9
2�/
)�&
input_1���������
p
� "%�"
�
0���������	
� �
C__inference_dqn_layer_call_and_return_conditional_losses_4374094449q6789:;<=>?@A&'./6�3
,�)
#� 
x���������
p 
� "%�"
�
0���������	
� �
C__inference_dqn_layer_call_and_return_conditional_losses_4374094523q6789:;<=>?@A&'./6�3
,�)
#� 
x���������
p
� "%�"
�
0���������	
� �
(__inference_dqn_layer_call_fn_4374093999j6789:;<=>?@A&'./<�9
2�/
)�&
input_1���������
p 
� "����������	�
(__inference_dqn_layer_call_fn_4374094191j6789:;<=>?@A&'./<�9
2�/
)�&
input_1���������
p
� "����������	�
(__inference_dqn_layer_call_fn_4374094338d6789:;<=>?@A&'./6�3
,�)
#� 
x���������
p 
� "����������	�
(__inference_dqn_layer_call_fn_4374094375d6789:;<=>?@A&'./6�3
,�)
#� 
x���������
p
� "����������	�
G__inference_flatten_layer_call_and_return_conditional_losses_4374094741`7�4
-�*
(�%
inputs���������
� "%�"
�
0���������@
� �
,__inference_flatten_layer_call_fn_4374094735S7�4
-�*
(�%
inputs���������
� "����������@�
G__inference_re_lu_1_layer_call_and_return_conditional_losses_4374094961h7�4
-�*
(�%
inputs���������
� "-�*
#� 
0���������
� �
,__inference_re_lu_1_layer_call_fn_4374094956[7�4
-�*
(�%
inputs���������
� " �����������
E__inference_re_lu_layer_call_and_return_conditional_losses_4374094870h7�4
-�*
(�%
inputs���������	
� "-�*
#� 
0���������	
� �
*__inference_re_lu_layer_call_fn_4374094865[7�4
-�*
(�%
inputs���������	
� " ����������	�
L__inference_sequential_1_layer_call_and_return_conditional_losses_4374093857�<=>?@AG�D
=�:
0�-
conv2d_1_input���������	
p 

 
� "-�*
#� 
0���������
� �
L__inference_sequential_1_layer_call_and_return_conditional_losses_4374093876�<=>?@AG�D
=�:
0�-
conv2d_1_input���������	
p

 
� "-�*
#� 
0���������
� �
L__inference_sequential_1_layer_call_and_return_conditional_losses_4374094705x<=>?@A?�<
5�2
(�%
inputs���������	
p 

 
� "-�*
#� 
0���������
� �
L__inference_sequential_1_layer_call_and_return_conditional_losses_4374094730x<=>?@A?�<
5�2
(�%
inputs���������	
p

 
� "-�*
#� 
0���������
� �
1__inference_sequential_1_layer_call_fn_4374093752s<=>?@AG�D
=�:
0�-
conv2d_1_input���������	
p 

 
� " �����������
1__inference_sequential_1_layer_call_fn_4374093838s<=>?@AG�D
=�:
0�-
conv2d_1_input���������	
p

 
� " �����������
1__inference_sequential_1_layer_call_fn_4374094663k<=>?@A?�<
5�2
(�%
inputs���������	
p 

 
� " �����������
1__inference_sequential_1_layer_call_fn_4374094680k<=>?@A?�<
5�2
(�%
inputs���������	
p

 
� " �����������
J__inference_sequential_layer_call_and_return_conditional_losses_4374093614~6789:;E�B
;�8
.�+
conv2d_input���������
p 

 
� "-�*
#� 
0���������	
� �
J__inference_sequential_layer_call_and_return_conditional_losses_4374093633~6789:;E�B
;�8
.�+
conv2d_input���������
p

 
� "-�*
#� 
0���������	
� �
J__inference_sequential_layer_call_and_return_conditional_losses_4374094621x6789:;?�<
5�2
(�%
inputs���������
p 

 
� "-�*
#� 
0���������	
� �
J__inference_sequential_layer_call_and_return_conditional_losses_4374094646x6789:;?�<
5�2
(�%
inputs���������
p

 
� "-�*
#� 
0���������	
� �
/__inference_sequential_layer_call_fn_4374093509q6789:;E�B
;�8
.�+
conv2d_input���������
p 

 
� " ����������	�
/__inference_sequential_layer_call_fn_4374093595q6789:;E�B
;�8
.�+
conv2d_input���������
p

 
� " ����������	�
/__inference_sequential_layer_call_fn_4374094579k6789:;?�<
5�2
(�%
inputs���������
p 

 
� " ����������	�
/__inference_sequential_layer_call_fn_4374094596k6789:;?�<
5�2
(�%
inputs���������
p

 
� " ����������	�
(__inference_signature_wrapper_4374094562�6789:;<=>?@A&'./C�@
� 
9�6
4
input_1)�&
input_1���������"3�0
.
output_1"�
output_1���������	