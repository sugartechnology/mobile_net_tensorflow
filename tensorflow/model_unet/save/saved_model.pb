ян%
¬У
B
AssignVariableOp
resource
value"dtype"
dtypetypeИ
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
Ы
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
ъ
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
epsilonfloat%Ј—8"&
exponential_avg_factorfloat%  А?";
data_formatstringNHWC:
NHWCNCHWNDHWCNCDHW"
is_trainingbool(
.
Identity

input"T
output"T"	
Ttype
В
MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 ":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(И
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
dtypetypeИ
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
Щ
ResizeNearestNeighbor
images"T
size
resized_images"T"
Ttype:
2
	"
align_cornersbool( "
half_pixel_centersbool( 
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
Ѕ
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
executor_typestring И®
@
StaticRegexFullMatch	
input

output
"
patternstring
ц
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
Ц
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 И"serve*2.7.02v2.7.0-rc1-69-gc256c071bb28ю≤!
~
conv2d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d/kernel
w
!conv2d/kernel/Read/ReadVariableOpReadVariableOpconv2d/kernel*&
_output_shapes
:*
dtype0
n
conv2d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d/bias
g
conv2d/bias/Read/ReadVariableOpReadVariableOpconv2d/bias*
_output_shapes
:*
dtype0
К
batch_normalization/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_namebatch_normalization/gamma
Г
-batch_normalization/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization/gamma*
_output_shapes
:*
dtype0
И
batch_normalization/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_namebatch_normalization/beta
Б
,batch_normalization/beta/Read/ReadVariableOpReadVariableOpbatch_normalization/beta*
_output_shapes
:*
dtype0
Ц
batch_normalization/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!batch_normalization/moving_mean
П
3batch_normalization/moving_mean/Read/ReadVariableOpReadVariableOpbatch_normalization/moving_mean*
_output_shapes
:*
dtype0
Ю
#batch_normalization/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#batch_normalization/moving_variance
Ч
7batch_normalization/moving_variance/Read/ReadVariableOpReadVariableOp#batch_normalization/moving_variance*
_output_shapes
:*
dtype0
В
conv2d_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: * 
shared_nameconv2d_1/kernel
{
#conv2d_1/kernel/Read/ReadVariableOpReadVariableOpconv2d_1/kernel*&
_output_shapes
: *
dtype0
r
conv2d_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d_1/bias
k
!conv2d_1/bias/Read/ReadVariableOpReadVariableOpconv2d_1/bias*
_output_shapes
: *
dtype0
О
batch_normalization_1/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *,
shared_namebatch_normalization_1/gamma
З
/batch_normalization_1/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_1/gamma*
_output_shapes
: *
dtype0
М
batch_normalization_1/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *+
shared_namebatch_normalization_1/beta
Е
.batch_normalization_1/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_1/beta*
_output_shapes
: *
dtype0
Ъ
!batch_normalization_1/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: *2
shared_name#!batch_normalization_1/moving_mean
У
5batch_normalization_1/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_1/moving_mean*
_output_shapes
: *
dtype0
Ґ
%batch_normalization_1/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: *6
shared_name'%batch_normalization_1/moving_variance
Ы
9batch_normalization_1/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_1/moving_variance*
_output_shapes
: *
dtype0
В
conv2d_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @* 
shared_nameconv2d_2/kernel
{
#conv2d_2/kernel/Read/ReadVariableOpReadVariableOpconv2d_2/kernel*&
_output_shapes
: @*
dtype0
r
conv2d_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_2/bias
k
!conv2d_2/bias/Read/ReadVariableOpReadVariableOpconv2d_2/bias*
_output_shapes
:@*
dtype0
О
batch_normalization_2/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_namebatch_normalization_2/gamma
З
/batch_normalization_2/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_2/gamma*
_output_shapes
:@*
dtype0
М
batch_normalization_2/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*+
shared_namebatch_normalization_2/beta
Е
.batch_normalization_2/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_2/beta*
_output_shapes
:@*
dtype0
Ъ
!batch_normalization_2/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*2
shared_name#!batch_normalization_2/moving_mean
У
5batch_normalization_2/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_2/moving_mean*
_output_shapes
:@*
dtype0
Ґ
%batch_normalization_2/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*6
shared_name'%batch_normalization_2/moving_variance
Ы
9batch_normalization_2/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_2/moving_variance*
_output_shapes
:@*
dtype0
Г
conv2d_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@А* 
shared_nameconv2d_3/kernel
|
#conv2d_3/kernel/Read/ReadVariableOpReadVariableOpconv2d_3/kernel*'
_output_shapes
:@А*
dtype0
s
conv2d_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*
shared_nameconv2d_3/bias
l
!conv2d_3/bias/Read/ReadVariableOpReadVariableOpconv2d_3/bias*
_output_shapes	
:А*
dtype0
П
batch_normalization_3/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*,
shared_namebatch_normalization_3/gamma
И
/batch_normalization_3/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_3/gamma*
_output_shapes	
:А*
dtype0
Н
batch_normalization_3/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*+
shared_namebatch_normalization_3/beta
Ж
.batch_normalization_3/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_3/beta*
_output_shapes	
:А*
dtype0
Ы
!batch_normalization_3/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*2
shared_name#!batch_normalization_3/moving_mean
Ф
5batch_normalization_3/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_3/moving_mean*
_output_shapes	
:А*
dtype0
£
%batch_normalization_3/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*6
shared_name'%batch_normalization_3/moving_variance
Ь
9batch_normalization_3/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_3/moving_variance*
_output_shapes	
:А*
dtype0
Д
conv2d_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:јј* 
shared_nameconv2d_4/kernel
}
#conv2d_4/kernel/Read/ReadVariableOpReadVariableOpconv2d_4/kernel*(
_output_shapes
:јј*
dtype0
s
conv2d_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:ј*
shared_nameconv2d_4/bias
l
!conv2d_4/bias/Read/ReadVariableOpReadVariableOpconv2d_4/bias*
_output_shapes	
:ј*
dtype0
П
batch_normalization_4/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:ј*,
shared_namebatch_normalization_4/gamma
И
/batch_normalization_4/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_4/gamma*
_output_shapes	
:ј*
dtype0
Н
batch_normalization_4/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:ј*+
shared_namebatch_normalization_4/beta
Ж
.batch_normalization_4/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_4/beta*
_output_shapes	
:ј*
dtype0
Ы
!batch_normalization_4/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:ј*2
shared_name#!batch_normalization_4/moving_mean
Ф
5batch_normalization_4/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_4/moving_mean*
_output_shapes	
:ј*
dtype0
£
%batch_normalization_4/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:ј*6
shared_name'%batch_normalization_4/moving_variance
Ь
9batch_normalization_4/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_4/moving_variance*
_output_shapes	
:ј*
dtype0
Г
conv2d_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:а`* 
shared_nameconv2d_5/kernel
|
#conv2d_5/kernel/Read/ReadVariableOpReadVariableOpconv2d_5/kernel*'
_output_shapes
:а`*
dtype0
r
conv2d_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:`*
shared_nameconv2d_5/bias
k
!conv2d_5/bias/Read/ReadVariableOpReadVariableOpconv2d_5/bias*
_output_shapes
:`*
dtype0
О
batch_normalization_5/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:`*,
shared_namebatch_normalization_5/gamma
З
/batch_normalization_5/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_5/gamma*
_output_shapes
:`*
dtype0
М
batch_normalization_5/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:`*+
shared_namebatch_normalization_5/beta
Е
.batch_normalization_5/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_5/beta*
_output_shapes
:`*
dtype0
Ъ
!batch_normalization_5/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:`*2
shared_name#!batch_normalization_5/moving_mean
У
5batch_normalization_5/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_5/moving_mean*
_output_shapes
:`*
dtype0
Ґ
%batch_normalization_5/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:`*6
shared_name'%batch_normalization_5/moving_variance
Ы
9batch_normalization_5/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_5/moving_variance*
_output_shapes
:`*
dtype0
В
conv2d_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:p0* 
shared_nameconv2d_6/kernel
{
#conv2d_6/kernel/Read/ReadVariableOpReadVariableOpconv2d_6/kernel*&
_output_shapes
:p0*
dtype0
r
conv2d_6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*
shared_nameconv2d_6/bias
k
!conv2d_6/bias/Read/ReadVariableOpReadVariableOpconv2d_6/bias*
_output_shapes
:0*
dtype0
О
batch_normalization_6/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*,
shared_namebatch_normalization_6/gamma
З
/batch_normalization_6/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_6/gamma*
_output_shapes
:0*
dtype0
М
batch_normalization_6/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*+
shared_namebatch_normalization_6/beta
Е
.batch_normalization_6/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_6/beta*
_output_shapes
:0*
dtype0
Ъ
!batch_normalization_6/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*2
shared_name#!batch_normalization_6/moving_mean
У
5batch_normalization_6/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_6/moving_mean*
_output_shapes
:0*
dtype0
Ґ
%batch_normalization_6/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*6
shared_name'%batch_normalization_6/moving_variance
Ы
9batch_normalization_6/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_6/moving_variance*
_output_shapes
:0*
dtype0
В
conv2d_7/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:0* 
shared_nameconv2d_7/kernel
{
#conv2d_7/kernel/Read/ReadVariableOpReadVariableOpconv2d_7/kernel*&
_output_shapes
:0*
dtype0
r
conv2d_7/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_7/bias
k
!conv2d_7/bias/Read/ReadVariableOpReadVariableOpconv2d_7/bias*
_output_shapes
:*
dtype0
В
conv2d/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv2d/kernel/m
{
#conv2d/kernel/m/Read/ReadVariableOpReadVariableOpconv2d/kernel/m*&
_output_shapes
:*
dtype0
r
conv2d/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d/bias/m
k
!conv2d/bias/m/Read/ReadVariableOpReadVariableOpconv2d/bias/m*
_output_shapes
:*
dtype0
О
batch_normalization/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization/gamma/m
З
/batch_normalization/gamma/m/Read/ReadVariableOpReadVariableOpbatch_normalization/gamma/m*
_output_shapes
:*
dtype0
М
batch_normalization/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_namebatch_normalization/beta/m
Е
.batch_normalization/beta/m/Read/ReadVariableOpReadVariableOpbatch_normalization/beta/m*
_output_shapes
:*
dtype0
Ж
conv2d_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *"
shared_nameconv2d_1/kernel/m

%conv2d_1/kernel/m/Read/ReadVariableOpReadVariableOpconv2d_1/kernel/m*&
_output_shapes
: *
dtype0
v
conv2d_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: * 
shared_nameconv2d_1/bias/m
o
#conv2d_1/bias/m/Read/ReadVariableOpReadVariableOpconv2d_1/bias/m*
_output_shapes
: *
dtype0
Т
batch_normalization_1/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *.
shared_namebatch_normalization_1/gamma/m
Л
1batch_normalization_1/gamma/m/Read/ReadVariableOpReadVariableOpbatch_normalization_1/gamma/m*
_output_shapes
: *
dtype0
Р
batch_normalization_1/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *-
shared_namebatch_normalization_1/beta/m
Й
0batch_normalization_1/beta/m/Read/ReadVariableOpReadVariableOpbatch_normalization_1/beta/m*
_output_shapes
: *
dtype0
Ж
conv2d_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*"
shared_nameconv2d_2/kernel/m

%conv2d_2/kernel/m/Read/ReadVariableOpReadVariableOpconv2d_2/kernel/m*&
_output_shapes
: @*
dtype0
v
conv2d_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@* 
shared_nameconv2d_2/bias/m
o
#conv2d_2/bias/m/Read/ReadVariableOpReadVariableOpconv2d_2/bias/m*
_output_shapes
:@*
dtype0
Т
batch_normalization_2/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*.
shared_namebatch_normalization_2/gamma/m
Л
1batch_normalization_2/gamma/m/Read/ReadVariableOpReadVariableOpbatch_normalization_2/gamma/m*
_output_shapes
:@*
dtype0
Р
batch_normalization_2/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*-
shared_namebatch_normalization_2/beta/m
Й
0batch_normalization_2/beta/m/Read/ReadVariableOpReadVariableOpbatch_normalization_2/beta/m*
_output_shapes
:@*
dtype0
З
conv2d_3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@А*"
shared_nameconv2d_3/kernel/m
А
%conv2d_3/kernel/m/Read/ReadVariableOpReadVariableOpconv2d_3/kernel/m*'
_output_shapes
:@А*
dtype0
w
conv2d_3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:А* 
shared_nameconv2d_3/bias/m
p
#conv2d_3/bias/m/Read/ReadVariableOpReadVariableOpconv2d_3/bias/m*
_output_shapes	
:А*
dtype0
У
batch_normalization_3/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*.
shared_namebatch_normalization_3/gamma/m
М
1batch_normalization_3/gamma/m/Read/ReadVariableOpReadVariableOpbatch_normalization_3/gamma/m*
_output_shapes	
:А*
dtype0
С
batch_normalization_3/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*-
shared_namebatch_normalization_3/beta/m
К
0batch_normalization_3/beta/m/Read/ReadVariableOpReadVariableOpbatch_normalization_3/beta/m*
_output_shapes	
:А*
dtype0
И
conv2d_4/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:јј*"
shared_nameconv2d_4/kernel/m
Б
%conv2d_4/kernel/m/Read/ReadVariableOpReadVariableOpconv2d_4/kernel/m*(
_output_shapes
:јј*
dtype0
w
conv2d_4/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:ј* 
shared_nameconv2d_4/bias/m
p
#conv2d_4/bias/m/Read/ReadVariableOpReadVariableOpconv2d_4/bias/m*
_output_shapes	
:ј*
dtype0
У
batch_normalization_4/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:ј*.
shared_namebatch_normalization_4/gamma/m
М
1batch_normalization_4/gamma/m/Read/ReadVariableOpReadVariableOpbatch_normalization_4/gamma/m*
_output_shapes	
:ј*
dtype0
С
batch_normalization_4/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:ј*-
shared_namebatch_normalization_4/beta/m
К
0batch_normalization_4/beta/m/Read/ReadVariableOpReadVariableOpbatch_normalization_4/beta/m*
_output_shapes	
:ј*
dtype0
З
conv2d_5/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:а`*"
shared_nameconv2d_5/kernel/m
А
%conv2d_5/kernel/m/Read/ReadVariableOpReadVariableOpconv2d_5/kernel/m*'
_output_shapes
:а`*
dtype0
v
conv2d_5/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:`* 
shared_nameconv2d_5/bias/m
o
#conv2d_5/bias/m/Read/ReadVariableOpReadVariableOpconv2d_5/bias/m*
_output_shapes
:`*
dtype0
Т
batch_normalization_5/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:`*.
shared_namebatch_normalization_5/gamma/m
Л
1batch_normalization_5/gamma/m/Read/ReadVariableOpReadVariableOpbatch_normalization_5/gamma/m*
_output_shapes
:`*
dtype0
Р
batch_normalization_5/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:`*-
shared_namebatch_normalization_5/beta/m
Й
0batch_normalization_5/beta/m/Read/ReadVariableOpReadVariableOpbatch_normalization_5/beta/m*
_output_shapes
:`*
dtype0
Ж
conv2d_6/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:p0*"
shared_nameconv2d_6/kernel/m

%conv2d_6/kernel/m/Read/ReadVariableOpReadVariableOpconv2d_6/kernel/m*&
_output_shapes
:p0*
dtype0
v
conv2d_6/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:0* 
shared_nameconv2d_6/bias/m
o
#conv2d_6/bias/m/Read/ReadVariableOpReadVariableOpconv2d_6/bias/m*
_output_shapes
:0*
dtype0
Т
batch_normalization_6/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*.
shared_namebatch_normalization_6/gamma/m
Л
1batch_normalization_6/gamma/m/Read/ReadVariableOpReadVariableOpbatch_normalization_6/gamma/m*
_output_shapes
:0*
dtype0
Р
batch_normalization_6/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*-
shared_namebatch_normalization_6/beta/m
Й
0batch_normalization_6/beta/m/Read/ReadVariableOpReadVariableOpbatch_normalization_6/beta/m*
_output_shapes
:0*
dtype0
Ж
conv2d_7/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*"
shared_nameconv2d_7/kernel/m

%conv2d_7/kernel/m/Read/ReadVariableOpReadVariableOpconv2d_7/kernel/m*&
_output_shapes
:0*
dtype0
v
conv2d_7/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv2d_7/bias/m
o
#conv2d_7/bias/m/Read/ReadVariableOpReadVariableOpconv2d_7/bias/m*
_output_shapes
:*
dtype0
В
conv2d/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv2d/kernel/v
{
#conv2d/kernel/v/Read/ReadVariableOpReadVariableOpconv2d/kernel/v*&
_output_shapes
:*
dtype0
r
conv2d/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d/bias/v
k
!conv2d/bias/v/Read/ReadVariableOpReadVariableOpconv2d/bias/v*
_output_shapes
:*
dtype0
О
batch_normalization/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization/gamma/v
З
/batch_normalization/gamma/v/Read/ReadVariableOpReadVariableOpbatch_normalization/gamma/v*
_output_shapes
:*
dtype0
М
batch_normalization/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_namebatch_normalization/beta/v
Е
.batch_normalization/beta/v/Read/ReadVariableOpReadVariableOpbatch_normalization/beta/v*
_output_shapes
:*
dtype0
Ж
conv2d_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *"
shared_nameconv2d_1/kernel/v

%conv2d_1/kernel/v/Read/ReadVariableOpReadVariableOpconv2d_1/kernel/v*&
_output_shapes
: *
dtype0
v
conv2d_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: * 
shared_nameconv2d_1/bias/v
o
#conv2d_1/bias/v/Read/ReadVariableOpReadVariableOpconv2d_1/bias/v*
_output_shapes
: *
dtype0
Т
batch_normalization_1/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *.
shared_namebatch_normalization_1/gamma/v
Л
1batch_normalization_1/gamma/v/Read/ReadVariableOpReadVariableOpbatch_normalization_1/gamma/v*
_output_shapes
: *
dtype0
Р
batch_normalization_1/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *-
shared_namebatch_normalization_1/beta/v
Й
0batch_normalization_1/beta/v/Read/ReadVariableOpReadVariableOpbatch_normalization_1/beta/v*
_output_shapes
: *
dtype0
Ж
conv2d_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*"
shared_nameconv2d_2/kernel/v

%conv2d_2/kernel/v/Read/ReadVariableOpReadVariableOpconv2d_2/kernel/v*&
_output_shapes
: @*
dtype0
v
conv2d_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@* 
shared_nameconv2d_2/bias/v
o
#conv2d_2/bias/v/Read/ReadVariableOpReadVariableOpconv2d_2/bias/v*
_output_shapes
:@*
dtype0
Т
batch_normalization_2/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*.
shared_namebatch_normalization_2/gamma/v
Л
1batch_normalization_2/gamma/v/Read/ReadVariableOpReadVariableOpbatch_normalization_2/gamma/v*
_output_shapes
:@*
dtype0
Р
batch_normalization_2/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*-
shared_namebatch_normalization_2/beta/v
Й
0batch_normalization_2/beta/v/Read/ReadVariableOpReadVariableOpbatch_normalization_2/beta/v*
_output_shapes
:@*
dtype0
З
conv2d_3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@А*"
shared_nameconv2d_3/kernel/v
А
%conv2d_3/kernel/v/Read/ReadVariableOpReadVariableOpconv2d_3/kernel/v*'
_output_shapes
:@А*
dtype0
w
conv2d_3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:А* 
shared_nameconv2d_3/bias/v
p
#conv2d_3/bias/v/Read/ReadVariableOpReadVariableOpconv2d_3/bias/v*
_output_shapes	
:А*
dtype0
У
batch_normalization_3/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*.
shared_namebatch_normalization_3/gamma/v
М
1batch_normalization_3/gamma/v/Read/ReadVariableOpReadVariableOpbatch_normalization_3/gamma/v*
_output_shapes	
:А*
dtype0
С
batch_normalization_3/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*-
shared_namebatch_normalization_3/beta/v
К
0batch_normalization_3/beta/v/Read/ReadVariableOpReadVariableOpbatch_normalization_3/beta/v*
_output_shapes	
:А*
dtype0
И
conv2d_4/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:јј*"
shared_nameconv2d_4/kernel/v
Б
%conv2d_4/kernel/v/Read/ReadVariableOpReadVariableOpconv2d_4/kernel/v*(
_output_shapes
:јј*
dtype0
w
conv2d_4/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:ј* 
shared_nameconv2d_4/bias/v
p
#conv2d_4/bias/v/Read/ReadVariableOpReadVariableOpconv2d_4/bias/v*
_output_shapes	
:ј*
dtype0
У
batch_normalization_4/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:ј*.
shared_namebatch_normalization_4/gamma/v
М
1batch_normalization_4/gamma/v/Read/ReadVariableOpReadVariableOpbatch_normalization_4/gamma/v*
_output_shapes	
:ј*
dtype0
С
batch_normalization_4/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:ј*-
shared_namebatch_normalization_4/beta/v
К
0batch_normalization_4/beta/v/Read/ReadVariableOpReadVariableOpbatch_normalization_4/beta/v*
_output_shapes	
:ј*
dtype0
З
conv2d_5/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:а`*"
shared_nameconv2d_5/kernel/v
А
%conv2d_5/kernel/v/Read/ReadVariableOpReadVariableOpconv2d_5/kernel/v*'
_output_shapes
:а`*
dtype0
v
conv2d_5/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:`* 
shared_nameconv2d_5/bias/v
o
#conv2d_5/bias/v/Read/ReadVariableOpReadVariableOpconv2d_5/bias/v*
_output_shapes
:`*
dtype0
Т
batch_normalization_5/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:`*.
shared_namebatch_normalization_5/gamma/v
Л
1batch_normalization_5/gamma/v/Read/ReadVariableOpReadVariableOpbatch_normalization_5/gamma/v*
_output_shapes
:`*
dtype0
Р
batch_normalization_5/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:`*-
shared_namebatch_normalization_5/beta/v
Й
0batch_normalization_5/beta/v/Read/ReadVariableOpReadVariableOpbatch_normalization_5/beta/v*
_output_shapes
:`*
dtype0
Ж
conv2d_6/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:p0*"
shared_nameconv2d_6/kernel/v

%conv2d_6/kernel/v/Read/ReadVariableOpReadVariableOpconv2d_6/kernel/v*&
_output_shapes
:p0*
dtype0
v
conv2d_6/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:0* 
shared_nameconv2d_6/bias/v
o
#conv2d_6/bias/v/Read/ReadVariableOpReadVariableOpconv2d_6/bias/v*
_output_shapes
:0*
dtype0
Т
batch_normalization_6/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*.
shared_namebatch_normalization_6/gamma/v
Л
1batch_normalization_6/gamma/v/Read/ReadVariableOpReadVariableOpbatch_normalization_6/gamma/v*
_output_shapes
:0*
dtype0
Р
batch_normalization_6/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*-
shared_namebatch_normalization_6/beta/v
Й
0batch_normalization_6/beta/v/Read/ReadVariableOpReadVariableOpbatch_normalization_6/beta/v*
_output_shapes
:0*
dtype0
Ж
conv2d_7/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*"
shared_nameconv2d_7/kernel/v

%conv2d_7/kernel/v/Read/ReadVariableOpReadVariableOpconv2d_7/kernel/v*&
_output_shapes
:0*
dtype0
v
conv2d_7/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv2d_7/bias/v
o
#conv2d_7/bias/v/Read/ReadVariableOpReadVariableOpconv2d_7/bias/v*
_output_shapes
:*
dtype0
O
ConstConst*
_output_shapes
:*
dtype0*
valueB:
Q
Const_1Const*
_output_shapes
:*
dtype0*
valueB: 
Q
Const_2Const*
_output_shapes
:*
dtype0*
valueB:@
R
Const_3Const*
_output_shapes
:*
dtype0*
valueB:А

NoOpNoOp
шљ
Const_4Const"/device:CPU:0*
_output_shapes
: *
dtype0*∞љ
value•љB°љ BЩљ
№
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer-4
layer_with_weights-2
layer-5
layer_with_weights-3
layer-6
layer-7
	layer-8

layer_with_weights-4

layer-9
layer_with_weights-5
layer-10
layer-11
layer-12
layer_with_weights-6
layer-13
layer_with_weights-7
layer-14
layer-15
layer-16
layer-17
layer-18
layer_with_weights-8
layer-19
layer_with_weights-9
layer-20
layer-21
layer-22
layer-23
layer_with_weights-10
layer-24
layer_with_weights-11
layer-25
layer-26
layer-27
layer-28
layer_with_weights-12
layer-29
layer_with_weights-13
layer-30
 layer-31
!layer-32
"layer_with_weights-14
"layer-33
#layer-34
$	optimizer
%	variables
&trainable_variables
'regularization_losses
(	keras_api
)
signatures
 
h

*kernel
+bias
,	variables
-trainable_variables
.regularization_losses
/	keras_api
Ч
0axis
	1gamma
2beta
3moving_mean
4moving_variance
5	variables
6trainable_variables
7regularization_losses
8	keras_api
R
9	variables
:trainable_variables
;regularization_losses
<	keras_api
R
=	variables
>trainable_variables
?regularization_losses
@	keras_api
h

Akernel
Bbias
C	variables
Dtrainable_variables
Eregularization_losses
F	keras_api
Ч
Gaxis
	Hgamma
Ibeta
Jmoving_mean
Kmoving_variance
L	variables
Mtrainable_variables
Nregularization_losses
O	keras_api
R
P	variables
Qtrainable_variables
Rregularization_losses
S	keras_api
R
T	variables
Utrainable_variables
Vregularization_losses
W	keras_api
h

Xkernel
Ybias
Z	variables
[trainable_variables
\regularization_losses
]	keras_api
Ч
^axis
	_gamma
`beta
amoving_mean
bmoving_variance
c	variables
dtrainable_variables
eregularization_losses
f	keras_api
R
g	variables
htrainable_variables
iregularization_losses
j	keras_api
R
k	variables
ltrainable_variables
mregularization_losses
n	keras_api
h

okernel
pbias
q	variables
rtrainable_variables
sregularization_losses
t	keras_api
Ч
uaxis
	vgamma
wbeta
xmoving_mean
ymoving_variance
z	variables
{trainable_variables
|regularization_losses
}	keras_api
T
~	variables
trainable_variables
Аregularization_losses
Б	keras_api
V
В	variables
Гtrainable_variables
Дregularization_losses
Е	keras_api

Ж	keras_api

З	keras_api
n
Иkernel
	Йbias
К	variables
Лtrainable_variables
Мregularization_losses
Н	keras_api
†
	Оaxis

Пgamma
	Рbeta
Сmoving_mean
Тmoving_variance
У	variables
Фtrainable_variables
Хregularization_losses
Ц	keras_api
V
Ч	variables
Шtrainable_variables
Щregularization_losses
Ъ	keras_api

Ы	keras_api

Ь	keras_api
n
Эkernel
	Юbias
Я	variables
†trainable_variables
°regularization_losses
Ґ	keras_api
†
	£axis

§gamma
	•beta
¶moving_mean
Іmoving_variance
®	variables
©trainable_variables
™regularization_losses
Ђ	keras_api
V
ђ	variables
≠trainable_variables
Ѓregularization_losses
ѓ	keras_api

∞	keras_api

±	keras_api
n
≤kernel
	≥bias
і	variables
µtrainable_variables
ґregularization_losses
Ј	keras_api
†
	Єaxis

єgamma
	Їbeta
їmoving_mean
Љmoving_variance
љ	variables
Њtrainable_variables
њregularization_losses
ј	keras_api
V
Ѕ	variables
¬trainable_variables
√regularization_losses
ƒ	keras_api

≈	keras_api
n
∆kernel
	«bias
»	variables
…trainable_variables
 regularization_losses
Ћ	keras_api
V
ћ	variables
Ќtrainable_variables
ќregularization_losses
ѕ	keras_api
ф*m№+mЁ1mё2mяAmаBmбHmвImгXmдYmе_mж`mзomиpmйvmкwmл	Иmм	Йmн	Пmо	Рmп	Эmр	Юmс	§mт	•mу	≤mф	≥mх	єmц	Їmч	∆mш	«mщ*vъ+vы1vь2vэAvюBv€HvАIvБXvВYvГ_vД`vЕovЖpvЗvvИwvЙ	ИvК	ЙvЛ	ПvМ	РvН	ЭvО	ЮvП	§vР	•vС	≤vТ	≥vУ	єvФ	ЇvХ	∆vЦ	«vЧ
к
*0
+1
12
23
34
45
A6
B7
H8
I9
J10
K11
X12
Y13
_14
`15
a16
b17
o18
p19
v20
w21
x22
y23
И24
Й25
П26
Р27
С28
Т29
Э30
Ю31
§32
•33
¶34
І35
≤36
≥37
є38
Ї39
ї40
Љ41
∆42
«43
ф
*0
+1
12
23
A4
B5
H6
I7
X8
Y9
_10
`11
o12
p13
v14
w15
И16
Й17
П18
Р19
Э20
Ю21
§22
•23
≤24
≥25
є26
Ї27
∆28
«29
 
≤
–non_trainable_variables
—layers
“metrics
 ”layer_regularization_losses
‘layer_metrics
%	variables
&trainable_variables
'regularization_losses
 
YW
VARIABLE_VALUEconv2d/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEconv2d/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

*0
+1

*0
+1
 
≤
’non_trainable_variables
÷layers
„metrics
 Ўlayer_regularization_losses
ўlayer_metrics
,	variables
-trainable_variables
.regularization_losses
 
db
VARIABLE_VALUEbatch_normalization/gamma5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUEbatch_normalization/beta4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUEbatch_normalization/moving_mean;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUE#batch_normalization/moving_variance?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

10
21
32
43

10
21
 
≤
Џnon_trainable_variables
џlayers
№metrics
 Ёlayer_regularization_losses
ёlayer_metrics
5	variables
6trainable_variables
7regularization_losses
 
 
 
≤
яnon_trainable_variables
аlayers
бmetrics
 вlayer_regularization_losses
гlayer_metrics
9	variables
:trainable_variables
;regularization_losses
 
 
 
≤
дnon_trainable_variables
еlayers
жmetrics
 зlayer_regularization_losses
иlayer_metrics
=	variables
>trainable_variables
?regularization_losses
[Y
VARIABLE_VALUEconv2d_1/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv2d_1/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

A0
B1

A0
B1
 
≤
йnon_trainable_variables
кlayers
лmetrics
 мlayer_regularization_losses
нlayer_metrics
C	variables
Dtrainable_variables
Eregularization_losses
 
fd
VARIABLE_VALUEbatch_normalization_1/gamma5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUEbatch_normalization_1/beta4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE!batch_normalization_1/moving_mean;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE%batch_normalization_1/moving_variance?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

H0
I1
J2
K3

H0
I1
 
≤
оnon_trainable_variables
пlayers
рmetrics
 сlayer_regularization_losses
тlayer_metrics
L	variables
Mtrainable_variables
Nregularization_losses
 
 
 
≤
уnon_trainable_variables
фlayers
хmetrics
 цlayer_regularization_losses
чlayer_metrics
P	variables
Qtrainable_variables
Rregularization_losses
 
 
 
≤
шnon_trainable_variables
щlayers
ъmetrics
 ыlayer_regularization_losses
ьlayer_metrics
T	variables
Utrainable_variables
Vregularization_losses
[Y
VARIABLE_VALUEconv2d_2/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv2d_2/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE

X0
Y1

X0
Y1
 
≤
эnon_trainable_variables
юlayers
€metrics
 Аlayer_regularization_losses
Бlayer_metrics
Z	variables
[trainable_variables
\regularization_losses
 
fd
VARIABLE_VALUEbatch_normalization_2/gamma5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUEbatch_normalization_2/beta4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE!batch_normalization_2/moving_mean;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE%batch_normalization_2/moving_variance?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

_0
`1
a2
b3

_0
`1
 
≤
Вnon_trainable_variables
Гlayers
Дmetrics
 Еlayer_regularization_losses
Жlayer_metrics
c	variables
dtrainable_variables
eregularization_losses
 
 
 
≤
Зnon_trainable_variables
Иlayers
Йmetrics
 Кlayer_regularization_losses
Лlayer_metrics
g	variables
htrainable_variables
iregularization_losses
 
 
 
≤
Мnon_trainable_variables
Нlayers
Оmetrics
 Пlayer_regularization_losses
Рlayer_metrics
k	variables
ltrainable_variables
mregularization_losses
[Y
VARIABLE_VALUEconv2d_3/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv2d_3/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE

o0
p1

o0
p1
 
≤
Сnon_trainable_variables
Тlayers
Уmetrics
 Фlayer_regularization_losses
Хlayer_metrics
q	variables
rtrainable_variables
sregularization_losses
 
fd
VARIABLE_VALUEbatch_normalization_3/gamma5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUEbatch_normalization_3/beta4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE!batch_normalization_3/moving_mean;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE%batch_normalization_3/moving_variance?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

v0
w1
x2
y3

v0
w1
 
≤
Цnon_trainable_variables
Чlayers
Шmetrics
 Щlayer_regularization_losses
Ъlayer_metrics
z	variables
{trainable_variables
|regularization_losses
 
 
 
≥
Ыnon_trainable_variables
Ьlayers
Эmetrics
 Юlayer_regularization_losses
Яlayer_metrics
~	variables
trainable_variables
Аregularization_losses
 
 
 
µ
†non_trainable_variables
°layers
Ґmetrics
 £layer_regularization_losses
§layer_metrics
В	variables
Гtrainable_variables
Дregularization_losses
 
 
[Y
VARIABLE_VALUEconv2d_4/kernel6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv2d_4/bias4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUE

И0
Й1

И0
Й1
 
µ
•non_trainable_variables
¶layers
Іmetrics
 ®layer_regularization_losses
©layer_metrics
К	variables
Лtrainable_variables
Мregularization_losses
 
fd
VARIABLE_VALUEbatch_normalization_4/gamma5layer_with_weights-9/gamma/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUEbatch_normalization_4/beta4layer_with_weights-9/beta/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE!batch_normalization_4/moving_mean;layer_with_weights-9/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE%batch_normalization_4/moving_variance?layer_with_weights-9/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 
П0
Р1
С2
Т3

П0
Р1
 
µ
™non_trainable_variables
Ђlayers
ђmetrics
 ≠layer_regularization_losses
Ѓlayer_metrics
У	variables
Фtrainable_variables
Хregularization_losses
 
 
 
µ
ѓnon_trainable_variables
∞layers
±metrics
 ≤layer_regularization_losses
≥layer_metrics
Ч	variables
Шtrainable_variables
Щregularization_losses
 
 
\Z
VARIABLE_VALUEconv2d_5/kernel7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_5/bias5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUE

Э0
Ю1

Э0
Ю1
 
µ
іnon_trainable_variables
µlayers
ґmetrics
 Јlayer_regularization_losses
Єlayer_metrics
Я	variables
†trainable_variables
°regularization_losses
 
ge
VARIABLE_VALUEbatch_normalization_5/gamma6layer_with_weights-11/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbatch_normalization_5/beta5layer_with_weights-11/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE!batch_normalization_5/moving_mean<layer_with_weights-11/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE%batch_normalization_5/moving_variance@layer_with_weights-11/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 
§0
•1
¶2
І3

§0
•1
 
µ
єnon_trainable_variables
Їlayers
їmetrics
 Љlayer_regularization_losses
љlayer_metrics
®	variables
©trainable_variables
™regularization_losses
 
 
 
µ
Њnon_trainable_variables
њlayers
јmetrics
 Ѕlayer_regularization_losses
¬layer_metrics
ђ	variables
≠trainable_variables
Ѓregularization_losses
 
 
\Z
VARIABLE_VALUEconv2d_6/kernel7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_6/bias5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUE

≤0
≥1

≤0
≥1
 
µ
√non_trainable_variables
ƒlayers
≈metrics
 ∆layer_regularization_losses
«layer_metrics
і	variables
µtrainable_variables
ґregularization_losses
 
ge
VARIABLE_VALUEbatch_normalization_6/gamma6layer_with_weights-13/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbatch_normalization_6/beta5layer_with_weights-13/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE!batch_normalization_6/moving_mean<layer_with_weights-13/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE%batch_normalization_6/moving_variance@layer_with_weights-13/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 
є0
Ї1
ї2
Љ3

є0
Ї1
 
µ
»non_trainable_variables
…layers
 metrics
 Ћlayer_regularization_losses
ћlayer_metrics
љ	variables
Њtrainable_variables
њregularization_losses
 
 
 
µ
Ќnon_trainable_variables
ќlayers
ѕmetrics
 –layer_regularization_losses
—layer_metrics
Ѕ	variables
¬trainable_variables
√regularization_losses
 
\Z
VARIABLE_VALUEconv2d_7/kernel7layer_with_weights-14/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_7/bias5layer_with_weights-14/bias/.ATTRIBUTES/VARIABLE_VALUE

∆0
«1

∆0
«1
 
µ
“non_trainable_variables
”layers
‘metrics
 ’layer_regularization_losses
÷layer_metrics
»	variables
…trainable_variables
 regularization_losses
 
 
 
µ
„non_trainable_variables
Ўlayers
ўmetrics
 Џlayer_regularization_losses
џlayer_metrics
ћ	variables
Ќtrainable_variables
ќregularization_losses
l
30
41
J2
K3
a4
b5
x6
y7
С8
Т9
¶10
І11
ї12
Љ13
О
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
30
 31
!32
"33
#34
 
 
 
 
 
 
 
 

30
41
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 

J0
K1
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 

a0
b1
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 

x0
y1
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 

С0
Т1
 
 
 
 
 
 
 
 
 
 
 
 
 
 

¶0
І1
 
 
 
 
 
 
 
 
 
 
 
 
 
 

ї0
Љ1
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
wu
VARIABLE_VALUEconv2d/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUEconv2d/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ГА
VARIABLE_VALUEbatch_normalization/gamma/mQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
А~
VARIABLE_VALUEbatch_normalization/beta/mPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEconv2d_1/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUEconv2d_1/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ЕВ
VARIABLE_VALUEbatch_normalization_1/gamma/mQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ГА
VARIABLE_VALUEbatch_normalization_1/beta/mPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEconv2d_2/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUEconv2d_2/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ЕВ
VARIABLE_VALUEbatch_normalization_2/gamma/mQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ГА
VARIABLE_VALUEbatch_normalization_2/beta/mPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEconv2d_3/kernel/mRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUEconv2d_3/bias/mPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ЕВ
VARIABLE_VALUEbatch_normalization_3/gamma/mQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ГА
VARIABLE_VALUEbatch_normalization_3/beta/mPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEconv2d_4/kernel/mRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUEconv2d_4/bias/mPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ЕВ
VARIABLE_VALUEbatch_normalization_4/gamma/mQlayer_with_weights-9/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ГА
VARIABLE_VALUEbatch_normalization_4/beta/mPlayer_with_weights-9/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEconv2d_5/kernel/mSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEconv2d_5/bias/mQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ЖГ
VARIABLE_VALUEbatch_normalization_5/gamma/mRlayer_with_weights-11/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ДБ
VARIABLE_VALUEbatch_normalization_5/beta/mQlayer_with_weights-11/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEconv2d_6/kernel/mSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEconv2d_6/bias/mQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ЖГ
VARIABLE_VALUEbatch_normalization_6/gamma/mRlayer_with_weights-13/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ДБ
VARIABLE_VALUEbatch_normalization_6/beta/mQlayer_with_weights-13/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEconv2d_7/kernel/mSlayer_with_weights-14/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEconv2d_7/bias/mQlayer_with_weights-14/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEconv2d/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUEconv2d/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ГА
VARIABLE_VALUEbatch_normalization/gamma/vQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
А~
VARIABLE_VALUEbatch_normalization/beta/vPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEconv2d_1/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUEconv2d_1/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ЕВ
VARIABLE_VALUEbatch_normalization_1/gamma/vQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ГА
VARIABLE_VALUEbatch_normalization_1/beta/vPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEconv2d_2/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUEconv2d_2/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ЕВ
VARIABLE_VALUEbatch_normalization_2/gamma/vQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ГА
VARIABLE_VALUEbatch_normalization_2/beta/vPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEconv2d_3/kernel/vRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUEconv2d_3/bias/vPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ЕВ
VARIABLE_VALUEbatch_normalization_3/gamma/vQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ГА
VARIABLE_VALUEbatch_normalization_3/beta/vPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEconv2d_4/kernel/vRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUEconv2d_4/bias/vPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ЕВ
VARIABLE_VALUEbatch_normalization_4/gamma/vQlayer_with_weights-9/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ГА
VARIABLE_VALUEbatch_normalization_4/beta/vPlayer_with_weights-9/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEconv2d_5/kernel/vSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEconv2d_5/bias/vQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ЖГ
VARIABLE_VALUEbatch_normalization_5/gamma/vRlayer_with_weights-11/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ДБ
VARIABLE_VALUEbatch_normalization_5/beta/vQlayer_with_weights-11/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEconv2d_6/kernel/vSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEconv2d_6/bias/vQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ЖГ
VARIABLE_VALUEbatch_normalization_6/gamma/vRlayer_with_weights-13/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ДБ
VARIABLE_VALUEbatch_normalization_6/beta/vQlayer_with_weights-13/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEconv2d_7/kernel/vSlayer_with_weights-14/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEconv2d_7/bias/vQlayer_with_weights-14/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
О
serving_default_input_1Placeholder*1
_output_shapes
:€€€€€€€€€АА*
dtype0*&
shape:€€€€€€€€€АА
ƒ
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1conv2d/kernelconv2d/biasbatch_normalization/gammabatch_normalization/betabatch_normalization/moving_mean#batch_normalization/moving_varianceconv2d_1/kernelconv2d_1/biasbatch_normalization_1/gammabatch_normalization_1/beta!batch_normalization_1/moving_mean%batch_normalization_1/moving_varianceconv2d_2/kernelconv2d_2/biasbatch_normalization_2/gammabatch_normalization_2/beta!batch_normalization_2/moving_mean%batch_normalization_2/moving_varianceconv2d_3/kernelconv2d_3/biasbatch_normalization_3/gammabatch_normalization_3/beta!batch_normalization_3/moving_mean%batch_normalization_3/moving_varianceConstconv2d_4/kernelconv2d_4/biasbatch_normalization_4/gammabatch_normalization_4/beta!batch_normalization_4/moving_mean%batch_normalization_4/moving_varianceConst_1conv2d_5/kernelconv2d_5/biasbatch_normalization_5/gammabatch_normalization_5/beta!batch_normalization_5/moving_mean%batch_normalization_5/moving_varianceConst_2conv2d_6/kernelconv2d_6/biasbatch_normalization_6/gammabatch_normalization_6/beta!batch_normalization_6/moving_mean%batch_normalization_6/moving_varianceConst_3conv2d_7/kernelconv2d_7/bias*<
Tin5
321*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:€€€€€€€€€АА*N
_read_only_resource_inputs0
.,	
!"#$%&()*+,-/0*0
config_proto 

CPU

GPU2*0J 8В *+
f&R$
"__inference_signature_wrapper_3471
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
µ'
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename!conv2d/kernel/Read/ReadVariableOpconv2d/bias/Read/ReadVariableOp-batch_normalization/gamma/Read/ReadVariableOp,batch_normalization/beta/Read/ReadVariableOp3batch_normalization/moving_mean/Read/ReadVariableOp7batch_normalization/moving_variance/Read/ReadVariableOp#conv2d_1/kernel/Read/ReadVariableOp!conv2d_1/bias/Read/ReadVariableOp/batch_normalization_1/gamma/Read/ReadVariableOp.batch_normalization_1/beta/Read/ReadVariableOp5batch_normalization_1/moving_mean/Read/ReadVariableOp9batch_normalization_1/moving_variance/Read/ReadVariableOp#conv2d_2/kernel/Read/ReadVariableOp!conv2d_2/bias/Read/ReadVariableOp/batch_normalization_2/gamma/Read/ReadVariableOp.batch_normalization_2/beta/Read/ReadVariableOp5batch_normalization_2/moving_mean/Read/ReadVariableOp9batch_normalization_2/moving_variance/Read/ReadVariableOp#conv2d_3/kernel/Read/ReadVariableOp!conv2d_3/bias/Read/ReadVariableOp/batch_normalization_3/gamma/Read/ReadVariableOp.batch_normalization_3/beta/Read/ReadVariableOp5batch_normalization_3/moving_mean/Read/ReadVariableOp9batch_normalization_3/moving_variance/Read/ReadVariableOp#conv2d_4/kernel/Read/ReadVariableOp!conv2d_4/bias/Read/ReadVariableOp/batch_normalization_4/gamma/Read/ReadVariableOp.batch_normalization_4/beta/Read/ReadVariableOp5batch_normalization_4/moving_mean/Read/ReadVariableOp9batch_normalization_4/moving_variance/Read/ReadVariableOp#conv2d_5/kernel/Read/ReadVariableOp!conv2d_5/bias/Read/ReadVariableOp/batch_normalization_5/gamma/Read/ReadVariableOp.batch_normalization_5/beta/Read/ReadVariableOp5batch_normalization_5/moving_mean/Read/ReadVariableOp9batch_normalization_5/moving_variance/Read/ReadVariableOp#conv2d_6/kernel/Read/ReadVariableOp!conv2d_6/bias/Read/ReadVariableOp/batch_normalization_6/gamma/Read/ReadVariableOp.batch_normalization_6/beta/Read/ReadVariableOp5batch_normalization_6/moving_mean/Read/ReadVariableOp9batch_normalization_6/moving_variance/Read/ReadVariableOp#conv2d_7/kernel/Read/ReadVariableOp!conv2d_7/bias/Read/ReadVariableOp#conv2d/kernel/m/Read/ReadVariableOp!conv2d/bias/m/Read/ReadVariableOp/batch_normalization/gamma/m/Read/ReadVariableOp.batch_normalization/beta/m/Read/ReadVariableOp%conv2d_1/kernel/m/Read/ReadVariableOp#conv2d_1/bias/m/Read/ReadVariableOp1batch_normalization_1/gamma/m/Read/ReadVariableOp0batch_normalization_1/beta/m/Read/ReadVariableOp%conv2d_2/kernel/m/Read/ReadVariableOp#conv2d_2/bias/m/Read/ReadVariableOp1batch_normalization_2/gamma/m/Read/ReadVariableOp0batch_normalization_2/beta/m/Read/ReadVariableOp%conv2d_3/kernel/m/Read/ReadVariableOp#conv2d_3/bias/m/Read/ReadVariableOp1batch_normalization_3/gamma/m/Read/ReadVariableOp0batch_normalization_3/beta/m/Read/ReadVariableOp%conv2d_4/kernel/m/Read/ReadVariableOp#conv2d_4/bias/m/Read/ReadVariableOp1batch_normalization_4/gamma/m/Read/ReadVariableOp0batch_normalization_4/beta/m/Read/ReadVariableOp%conv2d_5/kernel/m/Read/ReadVariableOp#conv2d_5/bias/m/Read/ReadVariableOp1batch_normalization_5/gamma/m/Read/ReadVariableOp0batch_normalization_5/beta/m/Read/ReadVariableOp%conv2d_6/kernel/m/Read/ReadVariableOp#conv2d_6/bias/m/Read/ReadVariableOp1batch_normalization_6/gamma/m/Read/ReadVariableOp0batch_normalization_6/beta/m/Read/ReadVariableOp%conv2d_7/kernel/m/Read/ReadVariableOp#conv2d_7/bias/m/Read/ReadVariableOp#conv2d/kernel/v/Read/ReadVariableOp!conv2d/bias/v/Read/ReadVariableOp/batch_normalization/gamma/v/Read/ReadVariableOp.batch_normalization/beta/v/Read/ReadVariableOp%conv2d_1/kernel/v/Read/ReadVariableOp#conv2d_1/bias/v/Read/ReadVariableOp1batch_normalization_1/gamma/v/Read/ReadVariableOp0batch_normalization_1/beta/v/Read/ReadVariableOp%conv2d_2/kernel/v/Read/ReadVariableOp#conv2d_2/bias/v/Read/ReadVariableOp1batch_normalization_2/gamma/v/Read/ReadVariableOp0batch_normalization_2/beta/v/Read/ReadVariableOp%conv2d_3/kernel/v/Read/ReadVariableOp#conv2d_3/bias/v/Read/ReadVariableOp1batch_normalization_3/gamma/v/Read/ReadVariableOp0batch_normalization_3/beta/v/Read/ReadVariableOp%conv2d_4/kernel/v/Read/ReadVariableOp#conv2d_4/bias/v/Read/ReadVariableOp1batch_normalization_4/gamma/v/Read/ReadVariableOp0batch_normalization_4/beta/v/Read/ReadVariableOp%conv2d_5/kernel/v/Read/ReadVariableOp#conv2d_5/bias/v/Read/ReadVariableOp1batch_normalization_5/gamma/v/Read/ReadVariableOp0batch_normalization_5/beta/v/Read/ReadVariableOp%conv2d_6/kernel/v/Read/ReadVariableOp#conv2d_6/bias/v/Read/ReadVariableOp1batch_normalization_6/gamma/v/Read/ReadVariableOp0batch_normalization_6/beta/v/Read/ReadVariableOp%conv2d_7/kernel/v/Read/ReadVariableOp#conv2d_7/bias/v/Read/ReadVariableOpConst_4*u
Tinn
l2j*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *&
f!R
__inference__traced_save_5574
О
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d/kernelconv2d/biasbatch_normalization/gammabatch_normalization/betabatch_normalization/moving_mean#batch_normalization/moving_varianceconv2d_1/kernelconv2d_1/biasbatch_normalization_1/gammabatch_normalization_1/beta!batch_normalization_1/moving_mean%batch_normalization_1/moving_varianceconv2d_2/kernelconv2d_2/biasbatch_normalization_2/gammabatch_normalization_2/beta!batch_normalization_2/moving_mean%batch_normalization_2/moving_varianceconv2d_3/kernelconv2d_3/biasbatch_normalization_3/gammabatch_normalization_3/beta!batch_normalization_3/moving_mean%batch_normalization_3/moving_varianceconv2d_4/kernelconv2d_4/biasbatch_normalization_4/gammabatch_normalization_4/beta!batch_normalization_4/moving_mean%batch_normalization_4/moving_varianceconv2d_5/kernelconv2d_5/biasbatch_normalization_5/gammabatch_normalization_5/beta!batch_normalization_5/moving_mean%batch_normalization_5/moving_varianceconv2d_6/kernelconv2d_6/biasbatch_normalization_6/gammabatch_normalization_6/beta!batch_normalization_6/moving_mean%batch_normalization_6/moving_varianceconv2d_7/kernelconv2d_7/biasconv2d/kernel/mconv2d/bias/mbatch_normalization/gamma/mbatch_normalization/beta/mconv2d_1/kernel/mconv2d_1/bias/mbatch_normalization_1/gamma/mbatch_normalization_1/beta/mconv2d_2/kernel/mconv2d_2/bias/mbatch_normalization_2/gamma/mbatch_normalization_2/beta/mconv2d_3/kernel/mconv2d_3/bias/mbatch_normalization_3/gamma/mbatch_normalization_3/beta/mconv2d_4/kernel/mconv2d_4/bias/mbatch_normalization_4/gamma/mbatch_normalization_4/beta/mconv2d_5/kernel/mconv2d_5/bias/mbatch_normalization_5/gamma/mbatch_normalization_5/beta/mconv2d_6/kernel/mconv2d_6/bias/mbatch_normalization_6/gamma/mbatch_normalization_6/beta/mconv2d_7/kernel/mconv2d_7/bias/mconv2d/kernel/vconv2d/bias/vbatch_normalization/gamma/vbatch_normalization/beta/vconv2d_1/kernel/vconv2d_1/bias/vbatch_normalization_1/gamma/vbatch_normalization_1/beta/vconv2d_2/kernel/vconv2d_2/bias/vbatch_normalization_2/gamma/vbatch_normalization_2/beta/vconv2d_3/kernel/vconv2d_3/bias/vbatch_normalization_3/gamma/vbatch_normalization_3/beta/vconv2d_4/kernel/vconv2d_4/bias/vbatch_normalization_4/gamma/vbatch_normalization_4/beta/vconv2d_5/kernel/vconv2d_5/bias/vbatch_normalization_5/gamma/vbatch_normalization_5/beta/vconv2d_6/kernel/vconv2d_6/bias/vbatch_normalization_6/gamma/vbatch_normalization_6/beta/vconv2d_7/kernel/vconv2d_7/bias/v*t
Tinm
k2i*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *)
f$R"
 __inference__traced_restore_5896„»
Џ
Ю
O__inference_batch_normalization_3_layer_call_and_return_conditional_losses_4653

inputs&
readvariableop_resource:	А(
readvariableop_1_resource:	А7
(fusedbatchnormv3_readvariableop_resource:	А9
*fusedbatchnormv3_readvariableop_1_resource:	А
identityИҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:А*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:А*
dtype0Е
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype0Й
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype0Ќ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А:А:А:А:А:*
epsilon%oГ:*
is_training( ~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А∞
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
 
_user_specified_nameinputs
«
ѕ
4__inference_batch_normalization_5_layer_call_fn_4948

inputs
unknown:`
	unknown_0:`
	unknown_1:`
	unknown_2:`
identityИҐStatefulPartitionedCallЙ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€  `*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *X
fSRQ
O__inference_batch_normalization_5_layer_call_and_return_conditional_losses_2004w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:€€€€€€€€€  ``
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:€€€€€€€€€  `: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:€€€€€€€€€  `
 
_user_specified_nameinputs
О
c
G__inference_max_pooling2d_layer_call_and_return_conditional_losses_4213

inputs
identity°
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€*
ksize
*
paddingSAME*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:r n
J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
…О
Ь
E__inference_Mobile_Unet_layer_call_and_return_conditional_losses_2111

inputs%
conv2d_1700:
conv2d_1702:&
batch_normalization_1723:&
batch_normalization_1725:&
batch_normalization_1727:&
batch_normalization_1729:'
conv2d_1_1756: 
conv2d_1_1758: (
batch_normalization_1_1779: (
batch_normalization_1_1781: (
batch_normalization_1_1783: (
batch_normalization_1_1785: '
conv2d_2_1812: @
conv2d_2_1814:@(
batch_normalization_2_1835:@(
batch_normalization_2_1837:@(
batch_normalization_2_1839:@(
batch_normalization_2_1841:@(
conv2d_3_1868:@А
conv2d_3_1870:	А)
batch_normalization_3_1891:	А)
batch_normalization_3_1893:	А)
batch_normalization_3_1895:	А)
batch_normalization_3_1897:	А
tf_image_resize_resize_size)
conv2d_4_1928:јј
conv2d_4_1930:	ј)
batch_normalization_4_1951:	ј)
batch_normalization_4_1953:	ј)
batch_normalization_4_1955:	ј)
batch_normalization_4_1957:	ј!
tf_image_resize_1_resize_size(
conv2d_5_1982:а`
conv2d_5_1984:`(
batch_normalization_5_2005:`(
batch_normalization_5_2007:`(
batch_normalization_5_2009:`(
batch_normalization_5_2011:`!
tf_image_resize_2_resize_size'
conv2d_6_2036:p0
conv2d_6_2038:0(
batch_normalization_6_2059:0(
batch_normalization_6_2061:0(
batch_normalization_6_2063:0(
batch_normalization_6_2065:0!
tf_image_resize_3_resize_size'
conv2d_7_2089:0
conv2d_7_2091:
identityИҐ+batch_normalization/StatefulPartitionedCallҐ-batch_normalization_1/StatefulPartitionedCallҐ-batch_normalization_2/StatefulPartitionedCallҐ-batch_normalization_3/StatefulPartitionedCallҐ-batch_normalization_4/StatefulPartitionedCallҐ-batch_normalization_5/StatefulPartitionedCallҐ-batch_normalization_6/StatefulPartitionedCallҐconv2d/StatefulPartitionedCallҐ conv2d_1/StatefulPartitionedCallҐ conv2d_2/StatefulPartitionedCallҐ conv2d_3/StatefulPartitionedCallҐ conv2d_4/StatefulPartitionedCallҐ conv2d_5/StatefulPartitionedCallҐ conv2d_6/StatefulPartitionedCallҐ conv2d_7/StatefulPartitionedCallп
conv2d/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_1700conv2d_1702*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:€€€€€€€€€АА*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *I
fDRB
@__inference_conv2d_layer_call_and_return_conditional_losses_1699ь
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0batch_normalization_1723batch_normalization_1725batch_normalization_1727batch_normalization_1729*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:€€€€€€€€€АА*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *V
fQRO
M__inference_batch_normalization_layer_call_and_return_conditional_losses_1722л
re_lu/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:€€€€€€€€€АА* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *H
fCRA
?__inference_re_lu_layer_call_and_return_conditional_losses_1737г
max_pooling2d/PartitionedCallPartitionedCallre_lu/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€@@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_max_pooling2d_layer_call_and_return_conditional_losses_1743Х
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0conv2d_1_1756conv2d_1_1758*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€@@ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_conv2d_1_layer_call_and_return_conditional_losses_1755И
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0batch_normalization_1_1779batch_normalization_1_1781batch_normalization_1_1783batch_normalization_1_1785*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€@@ *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *X
fSRQ
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1778п
re_lu_1/PartitionedCallPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€@@ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_re_lu_1_layer_call_and_return_conditional_losses_1793й
max_pooling2d_1/PartitionedCallPartitionedCall re_lu_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€   * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *R
fMRK
I__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_1799Ч
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_1/PartitionedCall:output:0conv2d_2_1812conv2d_2_1814*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€  @*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_conv2d_2_layer_call_and_return_conditional_losses_1811И
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0batch_normalization_2_1835batch_normalization_2_1837batch_normalization_2_1839batch_normalization_2_1841*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€  @*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *X
fSRQ
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1834п
re_lu_2/PartitionedCallPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€  @* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_re_lu_2_layer_call_and_return_conditional_losses_1849й
max_pooling2d_2/PartitionedCallPartitionedCall re_lu_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *R
fMRK
I__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_1855Ш
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_2/PartitionedCall:output:0conv2d_3_1868conv2d_3_1870*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_conv2d_3_layer_call_and_return_conditional_losses_1867Й
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0batch_normalization_3_1891batch_normalization_3_1893batch_normalization_3_1895batch_normalization_3_1897*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€А*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *X
fSRQ
O__inference_batch_normalization_3_layer_call_and_return_conditional_losses_1890р
re_lu_3/PartitionedCallPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_re_lu_3_layer_call_and_return_conditional_losses_1905к
max_pooling2d_3/PartitionedCallPartitionedCall re_lu_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *R
fMRK
I__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_1911б
,tf.image.resize/resize/ResizeNearestNeighborResizeNearestNeighbor(max_pooling2d_3/PartitionedCall:output:0tf_image_resize_resize_size*
T0*0
_output_shapes
:€€€€€€€€€А*
half_pixel_centers(W
tf.concat/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :й
tf.concat/concatConcatV2=tf.image.resize/resize/ResizeNearestNeighbor:resized_images:0(max_pooling2d_2/PartitionedCall:output:0tf.concat/concat/axis:output:0*
N*
T0*0
_output_shapes
:€€€€€€€€€јЙ
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCalltf.concat/concat:output:0conv2d_4_1928conv2d_4_1930*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€ј*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_conv2d_4_layer_call_and_return_conditional_losses_1927Й
-batch_normalization_4/StatefulPartitionedCallStatefulPartitionedCall)conv2d_4/StatefulPartitionedCall:output:0batch_normalization_4_1951batch_normalization_4_1953batch_normalization_4_1955batch_normalization_4_1957*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€ј*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *X
fSRQ
O__inference_batch_normalization_4_layer_call_and_return_conditional_losses_1950р
re_lu_4/PartitionedCallPartitionedCall6batch_normalization_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€ј* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_re_lu_4_layer_call_and_return_conditional_losses_1965Ё
.tf.image.resize_1/resize/ResizeNearestNeighborResizeNearestNeighbor re_lu_4/PartitionedCall:output:0tf_image_resize_1_resize_size*
T0*0
_output_shapes
:€€€€€€€€€  ј*
half_pixel_centers(Y
tf.concat_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :п
tf.concat_1/concatConcatV2?tf.image.resize_1/resize/ResizeNearestNeighbor:resized_images:0(max_pooling2d_1/PartitionedCall:output:0 tf.concat_1/concat/axis:output:0*
N*
T0*0
_output_shapes
:€€€€€€€€€  аК
 conv2d_5/StatefulPartitionedCallStatefulPartitionedCalltf.concat_1/concat:output:0conv2d_5_1982conv2d_5_1984*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€  `*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_conv2d_5_layer_call_and_return_conditional_losses_1981И
-batch_normalization_5/StatefulPartitionedCallStatefulPartitionedCall)conv2d_5/StatefulPartitionedCall:output:0batch_normalization_5_2005batch_normalization_5_2007batch_normalization_5_2009batch_normalization_5_2011*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€  `*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *X
fSRQ
O__inference_batch_normalization_5_layer_call_and_return_conditional_losses_2004п
re_lu_5/PartitionedCallPartitionedCall6batch_normalization_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€  `* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_re_lu_5_layer_call_and_return_conditional_losses_2019№
.tf.image.resize_2/resize/ResizeNearestNeighborResizeNearestNeighbor re_lu_5/PartitionedCall:output:0tf_image_resize_2_resize_size*
T0*/
_output_shapes
:€€€€€€€€€@@`*
half_pixel_centers(Y
tf.concat_2/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :м
tf.concat_2/concatConcatV2?tf.image.resize_2/resize/ResizeNearestNeighbor:resized_images:0&max_pooling2d/PartitionedCall:output:0 tf.concat_2/concat/axis:output:0*
N*
T0*/
_output_shapes
:€€€€€€€€€@@pК
 conv2d_6/StatefulPartitionedCallStatefulPartitionedCalltf.concat_2/concat:output:0conv2d_6_2036conv2d_6_2038*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€@@0*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_conv2d_6_layer_call_and_return_conditional_losses_2035И
-batch_normalization_6/StatefulPartitionedCallStatefulPartitionedCall)conv2d_6/StatefulPartitionedCall:output:0batch_normalization_6_2059batch_normalization_6_2061batch_normalization_6_2063batch_normalization_6_2065*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€@@0*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *X
fSRQ
O__inference_batch_normalization_6_layer_call_and_return_conditional_losses_2058п
re_lu_6/PartitionedCallPartitionedCall6batch_normalization_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€@@0* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_re_lu_6_layer_call_and_return_conditional_losses_2073ё
.tf.image.resize_3/resize/ResizeNearestNeighborResizeNearestNeighbor re_lu_6/PartitionedCall:output:0tf_image_resize_3_resize_size*
T0*1
_output_shapes
:€€€€€€€€€АА0*
half_pixel_centers(∞
 conv2d_7/StatefulPartitionedCallStatefulPartitionedCall?tf.image.resize_3/resize/ResizeNearestNeighbor:resized_images:0conv2d_7_2089conv2d_7_2091*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:€€€€€€€€€АА*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_conv2d_7_layer_call_and_return_conditional_losses_2088д
reshape/PartitionedCallPartitionedCall)conv2d_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:€€€€€€€€€АА* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_reshape_layer_call_and_return_conditional_losses_2108y
IdentityIdentity reshape/PartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:€€€€€€€€€АА™
NoOpNoOp,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall.^batch_normalization_3/StatefulPartitionedCall.^batch_normalization_4/StatefulPartitionedCall.^batch_normalization_5/StatefulPartitionedCall.^batch_normalization_6/StatefulPartitionedCall^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall!^conv2d_4/StatefulPartitionedCall!^conv2d_5/StatefulPartitionedCall!^conv2d_6/StatefulPartitionedCall!^conv2d_7/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Ґ
_input_shapesР
Н:€€€€€€€€€АА: : : : : : : : : : : : : : : : : : : : : : : : :: : : : : : :: : : : : : :: : : : : : :: : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2^
-batch_normalization_3/StatefulPartitionedCall-batch_normalization_3/StatefulPartitionedCall2^
-batch_normalization_4/StatefulPartitionedCall-batch_normalization_4/StatefulPartitionedCall2^
-batch_normalization_5/StatefulPartitionedCall-batch_normalization_5/StatefulPartitionedCall2^
-batch_normalization_6/StatefulPartitionedCall-batch_normalization_6/StatefulPartitionedCall2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall2D
 conv2d_5/StatefulPartitionedCall conv2d_5/StatefulPartitionedCall2D
 conv2d_6/StatefulPartitionedCall conv2d_6/StatefulPartitionedCall2D
 conv2d_7/StatefulPartitionedCall conv2d_7/StatefulPartitionedCall:Y U
1
_output_shapes
:€€€€€€€€€АА
 
_user_specified_nameinputs: 

_output_shapes
::  

_output_shapes
:: '

_output_shapes
:: .

_output_shapes
:
¶
c
G__inference_max_pooling2d_layer_call_and_return_conditional_losses_1743

inputs
identityЖ
MaxPoolMaxPoolinputs*/
_output_shapes
:€€€€€€€€€@@*
ksize
*
paddingSAME*
strides
`
IdentityIdentityMaxPool:output:0*
T0*/
_output_shapes
:€€€€€€€€€@@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:€€€€€€€€€АА:Y U
1
_output_shapes
:€€€€€€€€€АА
 
_user_specified_nameinputs
 
Ъ
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1284

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identityИҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0И
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0»
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ : : : : :*
epsilon%oГ:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ ∞
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ : : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 
 
_user_specified_nameinputs
•

ы
B__inference_conv2d_2_layer_call_and_return_conditional_losses_1811

inputs8
conv2d_readvariableop_resource: @-
biasadd_readvariableop_resource:@
identityИҐBiasAdd/ReadVariableOpҐConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0Щ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€  @*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€  @g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:€€€€€€€€€  @w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:€€€€€€€€€   : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:€€€€€€€€€   
 
_user_specified_nameinputs
…
Ќ
2__inference_batch_normalization_layer_call_fn_4116

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identityИҐStatefulPartitionedCallЗ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:€€€€€€€€€АА*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *V
fQRO
M__inference_batch_normalization_layer_call_and_return_conditional_losses_2645y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:€€€€€€€€€АА`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:€€€€€€€€€АА: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:€€€€€€€€€АА
 
_user_specified_nameinputs
Р	
ѕ
4__inference_batch_normalization_1_layer_call_fn_4250

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identityИҐStatefulPartitionedCallЫ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *X
fSRQ
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1284Й
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 
 
_user_specified_nameinputs
÷
Љ
M__inference_batch_normalization_layer_call_and_return_conditional_losses_4152

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identityИҐAssignNewValueҐAssignNewValue_1ҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0И
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0÷
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€:::::*
epsilon%oГ:*
exponential_avg_factor%
„#<∞
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0Ї
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€‘
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+€€€€€€€€€€€€€€€€€€€€€€€€€€€: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
и
¬
O__inference_batch_normalization_3_layer_call_and_return_conditional_losses_1467

inputs&
readvariableop_resource:	А(
readvariableop_1_resource:	А7
(fusedbatchnormv3_readvariableop_resource:	А9
*fusedbatchnormv3_readvariableop_1_resource:	А
identityИҐAssignNewValueҐAssignNewValue_1ҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:А*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:А*
dtype0Е
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype0Й
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype0џ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А:А:А:А:А:*
epsilon%oГ:*
exponential_avg_factor%
„#<∞
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0Ї
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А‘
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
 
_user_specified_nameinputs
Р
Њ
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_4534

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identityИҐAssignNewValueҐAssignNewValue_1ҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0И
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0ƒ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€  @:@:@:@:@:*
epsilon%oГ:*
exponential_avg_factor%
„#<∞
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0Ї
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0k
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:€€€€€€€€€  @‘
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:€€€€€€€€€  @: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:€€€€€€€€€  @
 
_user_specified_nameinputs
Р
Њ
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_2515

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identityИҐAssignNewValueҐAssignNewValue_1ҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0И
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0ƒ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€  @:@:@:@:@:*
epsilon%oГ:*
exponential_avg_factor%
„#<∞
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0Ї
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0k
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:€€€€€€€€€  @‘
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:€€€€€€€€€  @: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:€€€€€€€€€  @
 
_user_specified_nameinputs
Ц
Љ
M__inference_batch_normalization_layer_call_and_return_conditional_losses_4188

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identityИҐAssignNewValueҐAssignNewValue_1ҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0И
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0∆
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:€€€€€€€€€АА:::::*
epsilon%oГ:*
exponential_avg_factor%
„#<∞
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0Ї
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0m
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*1
_output_shapes
:€€€€€€€€€АА‘
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:€€€€€€€€€АА: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Y U
1
_output_shapes
:€€€€€€€€€АА
 
_user_specified_nameinputs
И
Ш
M__inference_batch_normalization_layer_call_and_return_conditional_losses_1722

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identityИҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0И
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0Є
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:€€€€€€€€€АА:::::*
epsilon%oГ:*
is_training( m
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*1
_output_shapes
:€€€€€€€€€АА∞
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:€€€€€€€€€АА: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Y U
1
_output_shapes
:€€€€€€€€€АА
 
_user_specified_nameinputs
†
¬
O__inference_batch_normalization_3_layer_call_and_return_conditional_losses_4707

inputs&
readvariableop_resource:	А(
readvariableop_1_resource:	А7
(fusedbatchnormv3_readvariableop_resource:	А9
*fusedbatchnormv3_readvariableop_1_resource:	А
identityИҐAssignNewValueҐAssignNewValue_1ҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:А*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:А*
dtype0Е
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype0Й
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype0…
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:€€€€€€€€€А:А:А:А:А:*
epsilon%oГ:*
exponential_avg_factor%
„#<∞
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0Ї
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0l
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*0
_output_shapes
:€€€€€€€€€А‘
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:€€€€€€€€€А: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:X T
0
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
Ў
Њ
O__inference_batch_normalization_5_layer_call_and_return_conditional_losses_4997

inputs%
readvariableop_resource:`'
readvariableop_1_resource:`6
(fusedbatchnormv3_readvariableop_resource:`8
*fusedbatchnormv3_readvariableop_1_resource:`
identityИҐAssignNewValueҐAssignNewValue_1ҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:`*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:`*
dtype0Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:`*
dtype0И
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:`*
dtype0÷
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€`:`:`:`:`:*
epsilon%oГ:*
exponential_avg_factor%
„#<∞
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0Ї
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€`‘
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+€€€€€€€€€€€€€€€€€€€€€€€€€€€`: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€`
 
_user_specified_nameinputs
л
[
?__inference_re_lu_layer_call_and_return_conditional_losses_4198

inputs
identityP
ReluReluinputs*
T0*1
_output_shapes
:€€€€€€€€€ААd
IdentityIdentityRelu:activations:0*
T0*1
_output_shapes
:€€€€€€€€€АА"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:€€€€€€€€€АА:Y U
1
_output_shapes
:€€€€€€€€€АА
 
_user_specified_nameinputs
М	
Ќ
2__inference_batch_normalization_layer_call_fn_4077

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identityИҐStatefulPartitionedCallЩ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *V
fQRO
M__inference_batch_normalization_layer_call_and_return_conditional_losses_1208Й
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+€€€€€€€€€€€€€€€€€€€€€€€€€€€: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
§
e
I__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_1799

inputs
identityЖ
MaxPoolMaxPoolinputs*/
_output_shapes
:€€€€€€€€€   *
ksize
*
paddingSAME*
strides
`
IdentityIdentityMaxPool:output:0*
T0*/
_output_shapes
:€€€€€€€€€   "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:€€€€€€€€€@@ :W S
/
_output_shapes
:€€€€€€€€€@@ 
 
_user_specified_nameinputs
¶
c
G__inference_max_pooling2d_layer_call_and_return_conditional_losses_4218

inputs
identityЖ
MaxPoolMaxPoolinputs*/
_output_shapes
:€€€€€€€€€@@*
ksize
*
paddingSAME*
strides
`
IdentityIdentityMaxPool:output:0*
T0*/
_output_shapes
:€€€€€€€€€@@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:€€€€€€€€€АА:Y U
1
_output_shapes
:€€€€€€€€€АА
 
_user_specified_nameinputs
†
¬
O__inference_batch_normalization_3_layer_call_and_return_conditional_losses_2450

inputs&
readvariableop_resource:	А(
readvariableop_1_resource:	А7
(fusedbatchnormv3_readvariableop_resource:	А9
*fusedbatchnormv3_readvariableop_1_resource:	А
identityИҐAssignNewValueҐAssignNewValue_1ҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:А*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:А*
dtype0Е
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype0Й
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype0…
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:€€€€€€€€€А:А:А:А:А:*
epsilon%oГ:*
exponential_avg_factor%
„#<∞
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0Ї
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0l
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*0
_output_shapes
:€€€€€€€€€А‘
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:€€€€€€€€€А: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:X T
0
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
ѕ
”
4__inference_batch_normalization_3_layer_call_fn_4622

inputs
unknown:	А
	unknown_0:	А
	unknown_1:	А
	unknown_2:	А
identityИҐStatefulPartitionedCallК
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€А*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *X
fSRQ
O__inference_batch_normalization_3_layer_call_and_return_conditional_losses_1890x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:€€€€€€€€€А`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:€€€€€€€€€А: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
Р
Њ
O__inference_batch_normalization_6_layer_call_and_return_conditional_losses_5186

inputs%
readvariableop_resource:0'
readvariableop_1_resource:06
(fusedbatchnormv3_readvariableop_resource:08
*fusedbatchnormv3_readvariableop_1_resource:0
identityИҐAssignNewValueҐAssignNewValue_1ҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:0*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:0*
dtype0Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype0И
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype0ƒ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€@@0:0:0:0:0:*
epsilon%oГ:*
exponential_avg_factor%
„#<∞
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0Ї
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0k
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:€€€€€€€€€@@0‘
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:€€€€€€€€€@@0: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:€€€€€€€€€@@0
 
_user_specified_nameinputs
 
Ъ
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_4480

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identityИҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0И
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0»
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@:@:@:@:@:*
epsilon%oГ:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@∞
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs
и
¬
O__inference_batch_normalization_4_layer_call_and_return_conditional_losses_1543

inputs&
readvariableop_resource:	ј(
readvariableop_1_resource:	ј7
(fusedbatchnormv3_readvariableop_resource:	ј9
*fusedbatchnormv3_readvariableop_1_resource:	ј
identityИҐAssignNewValueҐAssignNewValue_1ҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:ј*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:ј*
dtype0Е
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:ј*
dtype0Й
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:ј*
dtype0џ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,€€€€€€€€€€€€€€€€€€€€€€€€€€€ј:ј:ј:ј:ј:*
epsilon%oГ:*
exponential_avg_factor%
„#<∞
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0Ї
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€ј‘
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,€€€€€€€€€€€€€€€€€€€€€€€€€€€ј: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€ј
 
_user_specified_nameinputs
м
Э
'__inference_conv2d_5_layer_call_fn_4899

inputs"
unknown:а`
	unknown_0:`
identityИҐStatefulPartitionedCallв
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€  `*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_conv2d_5_layer_call_and_return_conditional_losses_1981w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:€€€€€€€€€  ``
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :€€€€€€€€€  а: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:€€€€€€€€€  а
 
_user_specified_nameinputs
§
e
I__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_4564

inputs
identityЖ
MaxPoolMaxPoolinputs*/
_output_shapes
:€€€€€€€€€@*
ksize
*
paddingSAME*
strides
`
IdentityIdentityMaxPool:output:0*
T0*/
_output_shapes
:€€€€€€€€€@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:€€€€€€€€€  @:W S
/
_output_shapes
:€€€€€€€€€  @
 
_user_specified_nameinputs
 
Ъ
O__inference_batch_normalization_6_layer_call_and_return_conditional_losses_1640

inputs%
readvariableop_resource:0'
readvariableop_1_resource:06
(fusedbatchnormv3_readvariableop_resource:08
*fusedbatchnormv3_readvariableop_1_resource:0
identityИҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:0*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:0*
dtype0Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype0И
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype0»
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€0:0:0:0:0:*
epsilon%oГ:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€0∞
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+€€€€€€€€€€€€€€€€€€€€€€€€€€€0: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€0
 
_user_specified_nameinputs
¬
B
&__inference_reshape_layer_call_fn_5221

inputs
identityє
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:€€€€€€€€€АА* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_reshape_layer_call_and_return_conditional_losses_2108j
IdentityIdentityPartitionedCall:output:0*
T0*1
_output_shapes
:€€€€€€€€€АА"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:€€€€€€€€€АА:Y U
1
_output_shapes
:€€€€€€€€€АА
 
_user_specified_nameinputs
†
¬
O__inference_batch_normalization_4_layer_call_and_return_conditional_losses_2385

inputs&
readvariableop_resource:	ј(
readvariableop_1_resource:	ј7
(fusedbatchnormv3_readvariableop_resource:	ј9
*fusedbatchnormv3_readvariableop_1_resource:	ј
identityИҐAssignNewValueҐAssignNewValue_1ҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:ј*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:ј*
dtype0Е
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:ј*
dtype0Й
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:ј*
dtype0…
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:€€€€€€€€€ј:ј:ј:ј:ј:*
epsilon%oГ:*
exponential_avg_factor%
„#<∞
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0Ї
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0l
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*0
_output_shapes
:€€€€€€€€€ј‘
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:€€€€€€€€€ј: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:X T
0
_output_shapes
:€€€€€€€€€ј
 
_user_specified_nameinputs
 
J
.__inference_max_pooling2d_2_layer_call_fn_4554

inputs
identityњ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *R
fMRK
I__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_1855h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:€€€€€€€€€@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:€€€€€€€€€  @:W S
/
_output_shapes
:€€€€€€€€€  @
 
_user_specified_nameinputs
В
Ъ
O__inference_batch_normalization_6_layer_call_and_return_conditional_losses_5168

inputs%
readvariableop_resource:0'
readvariableop_1_resource:06
(fusedbatchnormv3_readvariableop_resource:08
*fusedbatchnormv3_readvariableop_1_resource:0
identityИҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:0*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:0*
dtype0Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype0И
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype0ґ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€@@0:0:0:0:0:*
epsilon%oГ:*
is_training( k
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:€€€€€€€€€@@0∞
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:€€€€€€€€€@@0: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:€€€€€€€€€@@0
 
_user_specified_nameinputs
∞

ю
B__inference_conv2d_4_layer_call_and_return_conditional_losses_1927

inputs:
conv2d_readvariableop_resource:јј.
biasadd_readvariableop_resource:	ј
identityИҐBiasAdd/ReadVariableOpҐConv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:јј*
dtype0Ъ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€ј*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:ј*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€јh
IdentityIdentityBiasAdd:output:0^NoOp*
T0*0
_output_shapes
:€€€€€€€€€јw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :€€€€€€€€€ј: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:€€€€€€€€€ј
 
_user_specified_nameinputs
Ц	
”
4__inference_batch_normalization_3_layer_call_fn_4609

inputs
unknown:	А
	unknown_0:	А
	unknown_1:	А
	unknown_2:	А
identityИҐStatefulPartitionedCallЪ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *X
fSRQ
O__inference_batch_normalization_3_layer_call_and_return_conditional_losses_1467К
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
 
_user_specified_nameinputs
О	
ѕ
4__inference_batch_normalization_5_layer_call_fn_4935

inputs
unknown:`
	unknown_0:`
	unknown_1:`
	unknown_2:`
identityИҐStatefulPartitionedCallЩ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€`*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *X
fSRQ
O__inference_batch_normalization_5_layer_call_and_return_conditional_losses_1607Й
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€``
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+€€€€€€€€€€€€€€€€€€€€€€€€€€€`: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€`
 
_user_specified_nameinputs
и
¬
O__inference_batch_normalization_3_layer_call_and_return_conditional_losses_4671

inputs&
readvariableop_resource:	А(
readvariableop_1_resource:	А7
(fusedbatchnormv3_readvariableop_resource:	А9
*fusedbatchnormv3_readvariableop_1_resource:	А
identityИҐAssignNewValueҐAssignNewValue_1ҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:А*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:А*
dtype0Е
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype0Й
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype0џ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А:А:А:А:А:*
epsilon%oГ:*
exponential_avg_factor%
„#<∞
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0Ї
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А‘
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
 
_user_specified_nameinputs
е
]
A__inference_re_lu_1_layer_call_and_return_conditional_losses_1793

inputs
identityN
ReluReluinputs*
T0*/
_output_shapes
:€€€€€€€€€@@ b
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:€€€€€€€€€@@ "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:€€€€€€€€€@@ :W S
/
_output_shapes
:€€€€€€€€€@@ 
 
_user_specified_nameinputs
Ш	
”
4__inference_batch_normalization_3_layer_call_fn_4596

inputs
unknown:	А
	unknown_0:	А
	unknown_1:	А
	unknown_2:	А
identityИҐStatefulPartitionedCallЬ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *X
fSRQ
O__inference_batch_normalization_3_layer_call_and_return_conditional_losses_1436К
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
 
_user_specified_nameinputs
§
e
I__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_4391

inputs
identityЖ
MaxPoolMaxPoolinputs*/
_output_shapes
:€€€€€€€€€   *
ksize
*
paddingSAME*
strides
`
IdentityIdentityMaxPool:output:0*
T0*/
_output_shapes
:€€€€€€€€€   "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:€€€€€€€€€@@ :W S
/
_output_shapes
:€€€€€€€€€@@ 
 
_user_specified_nameinputs
с
Ь
'__inference_conv2d_7_layer_call_fn_5205

inputs!
unknown:0
	unknown_0:
identityИҐStatefulPartitionedCallд
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:€€€€€€€€€АА*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_conv2d_7_layer_call_and_return_conditional_losses_2088y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:€€€€€€€€€АА`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:€€€€€€€€€АА0: : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:€€€€€€€€€АА0
 
_user_specified_nameinputs
в
]
A__inference_reshape_layer_call_and_return_conditional_losses_5235

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:—
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :R
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value
B :АR
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value
B :А©
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:n
ReshapeReshapeinputsReshape/shape:output:0*
T0*1
_output_shapes
:€€€€€€€€€ААb
IdentityIdentityReshape:output:0*
T0*1
_output_shapes
:€€€€€€€€€АА"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:€€€€€€€€€АА:Y U
1
_output_shapes
:€€€€€€€€€АА
 
_user_specified_nameinputs
Ј
J
.__inference_max_pooling2d_3_layer_call_fn_4722

inputs
identityЏ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *R
fMRK
I__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_1487Г
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:r n
J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
й
Ь
'__inference_conv2d_2_layer_call_fn_4400

inputs!
unknown: @
	unknown_0:@
identityИҐStatefulPartitionedCallв
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€  @*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_conv2d_2_layer_call_and_return_conditional_losses_1811w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:€€€€€€€€€  @`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:€€€€€€€€€   : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:€€€€€€€€€   
 
_user_specified_nameinputs
≈
э

*__inference_Mobile_Unet_layer_call_fn_2210
input_1!
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:#
	unknown_5: 
	unknown_6: 
	unknown_7: 
	unknown_8: 
	unknown_9: 

unknown_10: $

unknown_11: @

unknown_12:@

unknown_13:@

unknown_14:@

unknown_15:@

unknown_16:@%

unknown_17:@А

unknown_18:	А

unknown_19:	А

unknown_20:	А

unknown_21:	А

unknown_22:	А

unknown_23&

unknown_24:јј

unknown_25:	ј

unknown_26:	ј

unknown_27:	ј

unknown_28:	ј

unknown_29:	ј

unknown_30%

unknown_31:а`

unknown_32:`

unknown_33:`

unknown_34:`

unknown_35:`

unknown_36:`

unknown_37$

unknown_38:p0

unknown_39:0

unknown_40:0

unknown_41:0

unknown_42:0

unknown_43:0

unknown_44$

unknown_45:0

unknown_46:
identityИҐStatefulPartitionedCallя
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46*<
Tin5
321*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:€€€€€€€€€АА*N
_read_only_resource_inputs0
.,	
!"#$%&()*+,-/0*0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_Mobile_Unet_layer_call_and_return_conditional_losses_2111y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:€€€€€€€€€АА`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Ґ
_input_shapesР
Н:€€€€€€€€€АА: : : : : : : : : : : : : : : : : : : : : : : : :: : : : : : :: : : : : : :: : : : : : :: : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
1
_output_shapes
:€€€€€€€€€АА
!
_user_specified_name	input_1: 

_output_shapes
::  

_output_shapes
:: '

_output_shapes
:: .

_output_shapes
:
•

ы
B__inference_conv2d_1_layer_call_and_return_conditional_losses_1755

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identityИҐBiasAdd/ReadVariableOpҐConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0Щ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€@@ *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€@@ g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:€€€€€€€€€@@ w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:€€€€€€€€€@@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:€€€€€€€€€@@
 
_user_specified_nameinputs
Ў
Њ
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1315

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identityИҐAssignNewValueҐAssignNewValue_1ҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0И
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0÷
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ : : : : :*
epsilon%oГ:*
exponential_avg_factor%
„#<∞
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0Ї
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ ‘
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ : : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 
 
_user_specified_nameinputs
Т
Ю
O__inference_batch_normalization_3_layer_call_and_return_conditional_losses_1890

inputs&
readvariableop_resource:	А(
readvariableop_1_resource:	А7
(fusedbatchnormv3_readvariableop_resource:	А9
*fusedbatchnormv3_readvariableop_1_resource:	А
identityИҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:А*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:А*
dtype0Е
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype0Й
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype0ї
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:€€€€€€€€€А:А:А:А:А:*
epsilon%oГ:*
is_training( l
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*0
_output_shapes
:€€€€€€€€€А∞
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:€€€€€€€€€А: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:X T
0
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
 
Ъ
O__inference_batch_normalization_5_layer_call_and_return_conditional_losses_1576

inputs%
readvariableop_resource:`'
readvariableop_1_resource:`6
(fusedbatchnormv3_readvariableop_resource:`8
*fusedbatchnormv3_readvariableop_1_resource:`
identityИҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:`*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:`*
dtype0Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:`*
dtype0И
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:`*
dtype0»
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€`:`:`:`:`:*
epsilon%oГ:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€`∞
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+€€€€€€€€€€€€€€€€€€€€€€€€€€€`: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€`
 
_user_specified_nameinputs
Р
Њ
O__inference_batch_normalization_6_layer_call_and_return_conditional_losses_2265

inputs%
readvariableop_resource:0'
readvariableop_1_resource:06
(fusedbatchnormv3_readvariableop_resource:08
*fusedbatchnormv3_readvariableop_1_resource:0
identityИҐAssignNewValueҐAssignNewValue_1ҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:0*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:0*
dtype0Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype0И
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype0ƒ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€@@0:0:0:0:0:*
epsilon%oГ:*
exponential_avg_factor%
„#<∞
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0Ї
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0k
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:€€€€€€€€€@@0‘
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:€€€€€€€€€@@0: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:€€€€€€€€€@@0
 
_user_specified_nameinputs
в
]
A__inference_reshape_layer_call_and_return_conditional_losses_2108

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:—
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :R
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value
B :АR
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value
B :А©
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:n
ReshapeReshapeinputsReshape/shape:output:0*
T0*1
_output_shapes
:€€€€€€€€€ААb
IdentityIdentityReshape:output:0*
T0*1
_output_shapes
:€€€€€€€€€АА"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:€€€€€€€€€АА:Y U
1
_output_shapes
:€€€€€€€€€АА
 
_user_specified_nameinputs
е
]
A__inference_re_lu_5_layer_call_and_return_conditional_losses_2019

inputs
identityN
ReluReluinputs*
T0*/
_output_shapes
:€€€€€€€€€  `b
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:€€€€€€€€€  `"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:€€€€€€€€€  `:W S
/
_output_shapes
:€€€€€€€€€  `
 
_user_specified_nameinputs
н
Ъ
%__inference_conv2d_layer_call_fn_4054

inputs!
unknown:
	unknown_0:
identityИҐStatefulPartitionedCallв
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:€€€€€€€€€АА*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *I
fDRB
@__inference_conv2d_layer_call_and_return_conditional_losses_1699y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:€€€€€€€€€АА`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:€€€€€€€€€АА: : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:€€€€€€€€€АА
 
_user_specified_nameinputs
Р
Њ
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_2580

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identityИҐAssignNewValueҐAssignNewValue_1ҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0И
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0ƒ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€@@ : : : : :*
epsilon%oГ:*
exponential_avg_factor%
„#<∞
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0Ї
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0k
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:€€€€€€€€€@@ ‘
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:€€€€€€€€€@@ : : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:€€€€€€€€€@@ 
 
_user_specified_nameinputs
 
H
,__inference_max_pooling2d_layer_call_fn_4208

inputs
identityљ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€@@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_max_pooling2d_layer_call_and_return_conditional_losses_1743h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:€€€€€€€€€@@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:€€€€€€€€€АА:Y U
1
_output_shapes
:€€€€€€€€€АА
 
_user_specified_nameinputs
е
]
A__inference_re_lu_1_layer_call_and_return_conditional_losses_4371

inputs
identityN
ReluReluinputs*
T0*/
_output_shapes
:€€€€€€€€€@@ b
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:€€€€€€€€€@@ "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:€€€€€€€€€@@ :W S
/
_output_shapes
:€€€€€€€€€@@ 
 
_user_specified_nameinputs
«
ѕ
4__inference_batch_normalization_1_layer_call_fn_4276

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identityИҐStatefulPartitionedCallЙ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€@@ *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *X
fSRQ
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1778w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:€€€€€€€€€@@ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:€€€€€€€€€@@ : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:€€€€€€€€€@@ 
 
_user_specified_nameinputs
Њ
B
&__inference_re_lu_3_layer_call_fn_4712

inputs
identityЄ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_re_lu_3_layer_call_and_return_conditional_losses_1905i
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:€€€€€€€€€А"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:€€€€€€€€€А:X T
0
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
™Ћ
У.
__inference__traced_save_5574
file_prefix,
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
@savev2_batch_normalization_1_moving_variance_read_readvariableop.
*savev2_conv2d_2_kernel_read_readvariableop,
(savev2_conv2d_2_bias_read_readvariableop:
6savev2_batch_normalization_2_gamma_read_readvariableop9
5savev2_batch_normalization_2_beta_read_readvariableop@
<savev2_batch_normalization_2_moving_mean_read_readvariableopD
@savev2_batch_normalization_2_moving_variance_read_readvariableop.
*savev2_conv2d_3_kernel_read_readvariableop,
(savev2_conv2d_3_bias_read_readvariableop:
6savev2_batch_normalization_3_gamma_read_readvariableop9
5savev2_batch_normalization_3_beta_read_readvariableop@
<savev2_batch_normalization_3_moving_mean_read_readvariableopD
@savev2_batch_normalization_3_moving_variance_read_readvariableop.
*savev2_conv2d_4_kernel_read_readvariableop,
(savev2_conv2d_4_bias_read_readvariableop:
6savev2_batch_normalization_4_gamma_read_readvariableop9
5savev2_batch_normalization_4_beta_read_readvariableop@
<savev2_batch_normalization_4_moving_mean_read_readvariableopD
@savev2_batch_normalization_4_moving_variance_read_readvariableop.
*savev2_conv2d_5_kernel_read_readvariableop,
(savev2_conv2d_5_bias_read_readvariableop:
6savev2_batch_normalization_5_gamma_read_readvariableop9
5savev2_batch_normalization_5_beta_read_readvariableop@
<savev2_batch_normalization_5_moving_mean_read_readvariableopD
@savev2_batch_normalization_5_moving_variance_read_readvariableop.
*savev2_conv2d_6_kernel_read_readvariableop,
(savev2_conv2d_6_bias_read_readvariableop:
6savev2_batch_normalization_6_gamma_read_readvariableop9
5savev2_batch_normalization_6_beta_read_readvariableop@
<savev2_batch_normalization_6_moving_mean_read_readvariableopD
@savev2_batch_normalization_6_moving_variance_read_readvariableop.
*savev2_conv2d_7_kernel_read_readvariableop,
(savev2_conv2d_7_bias_read_readvariableop.
*savev2_conv2d_kernel_m_read_readvariableop,
(savev2_conv2d_bias_m_read_readvariableop:
6savev2_batch_normalization_gamma_m_read_readvariableop9
5savev2_batch_normalization_beta_m_read_readvariableop0
,savev2_conv2d_1_kernel_m_read_readvariableop.
*savev2_conv2d_1_bias_m_read_readvariableop<
8savev2_batch_normalization_1_gamma_m_read_readvariableop;
7savev2_batch_normalization_1_beta_m_read_readvariableop0
,savev2_conv2d_2_kernel_m_read_readvariableop.
*savev2_conv2d_2_bias_m_read_readvariableop<
8savev2_batch_normalization_2_gamma_m_read_readvariableop;
7savev2_batch_normalization_2_beta_m_read_readvariableop0
,savev2_conv2d_3_kernel_m_read_readvariableop.
*savev2_conv2d_3_bias_m_read_readvariableop<
8savev2_batch_normalization_3_gamma_m_read_readvariableop;
7savev2_batch_normalization_3_beta_m_read_readvariableop0
,savev2_conv2d_4_kernel_m_read_readvariableop.
*savev2_conv2d_4_bias_m_read_readvariableop<
8savev2_batch_normalization_4_gamma_m_read_readvariableop;
7savev2_batch_normalization_4_beta_m_read_readvariableop0
,savev2_conv2d_5_kernel_m_read_readvariableop.
*savev2_conv2d_5_bias_m_read_readvariableop<
8savev2_batch_normalization_5_gamma_m_read_readvariableop;
7savev2_batch_normalization_5_beta_m_read_readvariableop0
,savev2_conv2d_6_kernel_m_read_readvariableop.
*savev2_conv2d_6_bias_m_read_readvariableop<
8savev2_batch_normalization_6_gamma_m_read_readvariableop;
7savev2_batch_normalization_6_beta_m_read_readvariableop0
,savev2_conv2d_7_kernel_m_read_readvariableop.
*savev2_conv2d_7_bias_m_read_readvariableop.
*savev2_conv2d_kernel_v_read_readvariableop,
(savev2_conv2d_bias_v_read_readvariableop:
6savev2_batch_normalization_gamma_v_read_readvariableop9
5savev2_batch_normalization_beta_v_read_readvariableop0
,savev2_conv2d_1_kernel_v_read_readvariableop.
*savev2_conv2d_1_bias_v_read_readvariableop<
8savev2_batch_normalization_1_gamma_v_read_readvariableop;
7savev2_batch_normalization_1_beta_v_read_readvariableop0
,savev2_conv2d_2_kernel_v_read_readvariableop.
*savev2_conv2d_2_bias_v_read_readvariableop<
8savev2_batch_normalization_2_gamma_v_read_readvariableop;
7savev2_batch_normalization_2_beta_v_read_readvariableop0
,savev2_conv2d_3_kernel_v_read_readvariableop.
*savev2_conv2d_3_bias_v_read_readvariableop<
8savev2_batch_normalization_3_gamma_v_read_readvariableop;
7savev2_batch_normalization_3_beta_v_read_readvariableop0
,savev2_conv2d_4_kernel_v_read_readvariableop.
*savev2_conv2d_4_bias_v_read_readvariableop<
8savev2_batch_normalization_4_gamma_v_read_readvariableop;
7savev2_batch_normalization_4_beta_v_read_readvariableop0
,savev2_conv2d_5_kernel_v_read_readvariableop.
*savev2_conv2d_5_bias_v_read_readvariableop<
8savev2_batch_normalization_5_gamma_v_read_readvariableop;
7savev2_batch_normalization_5_beta_v_read_readvariableop0
,savev2_conv2d_6_kernel_v_read_readvariableop.
*savev2_conv2d_6_bias_v_read_readvariableop<
8savev2_batch_normalization_6_gamma_v_read_readvariableop;
7savev2_batch_normalization_6_beta_v_read_readvariableop0
,savev2_conv2d_7_kernel_v_read_readvariableop.
*savev2_conv2d_7_bias_v_read_readvariableop
savev2_const_4

identity_1ИҐMergeV2Checkpointsw
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
_temp/partБ
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
value	B : У
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: п;
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:i*
dtype0*Ш;
valueО;BЛ;iB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-9/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-9/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-9/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-11/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-11/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-11/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-13/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-13/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-13/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-13/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-14/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-14/bias/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-9/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-11/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-13/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-13/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-14/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-14/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-9/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-11/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-13/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-13/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-14/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-14/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH¬
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:i*
dtype0*з
valueЁBЏiB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B °,
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0(savev2_conv2d_kernel_read_readvariableop&savev2_conv2d_bias_read_readvariableop4savev2_batch_normalization_gamma_read_readvariableop3savev2_batch_normalization_beta_read_readvariableop:savev2_batch_normalization_moving_mean_read_readvariableop>savev2_batch_normalization_moving_variance_read_readvariableop*savev2_conv2d_1_kernel_read_readvariableop(savev2_conv2d_1_bias_read_readvariableop6savev2_batch_normalization_1_gamma_read_readvariableop5savev2_batch_normalization_1_beta_read_readvariableop<savev2_batch_normalization_1_moving_mean_read_readvariableop@savev2_batch_normalization_1_moving_variance_read_readvariableop*savev2_conv2d_2_kernel_read_readvariableop(savev2_conv2d_2_bias_read_readvariableop6savev2_batch_normalization_2_gamma_read_readvariableop5savev2_batch_normalization_2_beta_read_readvariableop<savev2_batch_normalization_2_moving_mean_read_readvariableop@savev2_batch_normalization_2_moving_variance_read_readvariableop*savev2_conv2d_3_kernel_read_readvariableop(savev2_conv2d_3_bias_read_readvariableop6savev2_batch_normalization_3_gamma_read_readvariableop5savev2_batch_normalization_3_beta_read_readvariableop<savev2_batch_normalization_3_moving_mean_read_readvariableop@savev2_batch_normalization_3_moving_variance_read_readvariableop*savev2_conv2d_4_kernel_read_readvariableop(savev2_conv2d_4_bias_read_readvariableop6savev2_batch_normalization_4_gamma_read_readvariableop5savev2_batch_normalization_4_beta_read_readvariableop<savev2_batch_normalization_4_moving_mean_read_readvariableop@savev2_batch_normalization_4_moving_variance_read_readvariableop*savev2_conv2d_5_kernel_read_readvariableop(savev2_conv2d_5_bias_read_readvariableop6savev2_batch_normalization_5_gamma_read_readvariableop5savev2_batch_normalization_5_beta_read_readvariableop<savev2_batch_normalization_5_moving_mean_read_readvariableop@savev2_batch_normalization_5_moving_variance_read_readvariableop*savev2_conv2d_6_kernel_read_readvariableop(savev2_conv2d_6_bias_read_readvariableop6savev2_batch_normalization_6_gamma_read_readvariableop5savev2_batch_normalization_6_beta_read_readvariableop<savev2_batch_normalization_6_moving_mean_read_readvariableop@savev2_batch_normalization_6_moving_variance_read_readvariableop*savev2_conv2d_7_kernel_read_readvariableop(savev2_conv2d_7_bias_read_readvariableop*savev2_conv2d_kernel_m_read_readvariableop(savev2_conv2d_bias_m_read_readvariableop6savev2_batch_normalization_gamma_m_read_readvariableop5savev2_batch_normalization_beta_m_read_readvariableop,savev2_conv2d_1_kernel_m_read_readvariableop*savev2_conv2d_1_bias_m_read_readvariableop8savev2_batch_normalization_1_gamma_m_read_readvariableop7savev2_batch_normalization_1_beta_m_read_readvariableop,savev2_conv2d_2_kernel_m_read_readvariableop*savev2_conv2d_2_bias_m_read_readvariableop8savev2_batch_normalization_2_gamma_m_read_readvariableop7savev2_batch_normalization_2_beta_m_read_readvariableop,savev2_conv2d_3_kernel_m_read_readvariableop*savev2_conv2d_3_bias_m_read_readvariableop8savev2_batch_normalization_3_gamma_m_read_readvariableop7savev2_batch_normalization_3_beta_m_read_readvariableop,savev2_conv2d_4_kernel_m_read_readvariableop*savev2_conv2d_4_bias_m_read_readvariableop8savev2_batch_normalization_4_gamma_m_read_readvariableop7savev2_batch_normalization_4_beta_m_read_readvariableop,savev2_conv2d_5_kernel_m_read_readvariableop*savev2_conv2d_5_bias_m_read_readvariableop8savev2_batch_normalization_5_gamma_m_read_readvariableop7savev2_batch_normalization_5_beta_m_read_readvariableop,savev2_conv2d_6_kernel_m_read_readvariableop*savev2_conv2d_6_bias_m_read_readvariableop8savev2_batch_normalization_6_gamma_m_read_readvariableop7savev2_batch_normalization_6_beta_m_read_readvariableop,savev2_conv2d_7_kernel_m_read_readvariableop*savev2_conv2d_7_bias_m_read_readvariableop*savev2_conv2d_kernel_v_read_readvariableop(savev2_conv2d_bias_v_read_readvariableop6savev2_batch_normalization_gamma_v_read_readvariableop5savev2_batch_normalization_beta_v_read_readvariableop,savev2_conv2d_1_kernel_v_read_readvariableop*savev2_conv2d_1_bias_v_read_readvariableop8savev2_batch_normalization_1_gamma_v_read_readvariableop7savev2_batch_normalization_1_beta_v_read_readvariableop,savev2_conv2d_2_kernel_v_read_readvariableop*savev2_conv2d_2_bias_v_read_readvariableop8savev2_batch_normalization_2_gamma_v_read_readvariableop7savev2_batch_normalization_2_beta_v_read_readvariableop,savev2_conv2d_3_kernel_v_read_readvariableop*savev2_conv2d_3_bias_v_read_readvariableop8savev2_batch_normalization_3_gamma_v_read_readvariableop7savev2_batch_normalization_3_beta_v_read_readvariableop,savev2_conv2d_4_kernel_v_read_readvariableop*savev2_conv2d_4_bias_v_read_readvariableop8savev2_batch_normalization_4_gamma_v_read_readvariableop7savev2_batch_normalization_4_beta_v_read_readvariableop,savev2_conv2d_5_kernel_v_read_readvariableop*savev2_conv2d_5_bias_v_read_readvariableop8savev2_batch_normalization_5_gamma_v_read_readvariableop7savev2_batch_normalization_5_beta_v_read_readvariableop,savev2_conv2d_6_kernel_v_read_readvariableop*savev2_conv2d_6_bias_v_read_readvariableop8savev2_batch_normalization_6_gamma_v_read_readvariableop7savev2_batch_normalization_6_beta_v_read_readvariableop,savev2_conv2d_7_kernel_v_read_readvariableop*savev2_conv2d_7_bias_v_read_readvariableopsavev2_const_4"/device:CPU:0*
_output_shapes
 *w
dtypesm
k2iР
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:Л
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

identity_1Identity_1:output:0*Ћ
_input_shapesє
ґ: ::::::: : : : : : : @:@:@:@:@:@:@А:А:А:А:А:А:јј:ј:ј:ј:ј:ј:а`:`:`:`:`:`:p0:0:0:0:0:0:0:::::: : : : : @:@:@:@:@А:А:А:А:јј:ј:ј:ј:а`:`:`:`:p0:0:0:0:0:::::: : : : : @:@:@:@:@А:А:А:А:јј:ј:ј:ј:а`:`:`:`:p0:0:0:0:0:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
: : 

_output_shapes
: : 	

_output_shapes
: : 


_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
: @: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@:-)
'
_output_shapes
:@А:!

_output_shapes	
:А:!

_output_shapes	
:А:!

_output_shapes	
:А:!

_output_shapes	
:А:!

_output_shapes	
:А:.*
(
_output_shapes
:јј:!

_output_shapes	
:ј:!

_output_shapes	
:ј:!

_output_shapes	
:ј:!

_output_shapes	
:ј:!

_output_shapes	
:ј:-)
'
_output_shapes
:а`:  

_output_shapes
:`: !

_output_shapes
:`: "

_output_shapes
:`: #

_output_shapes
:`: $

_output_shapes
:`:,%(
&
_output_shapes
:p0: &

_output_shapes
:0: '

_output_shapes
:0: (

_output_shapes
:0: )

_output_shapes
:0: *

_output_shapes
:0:,+(
&
_output_shapes
:0: ,

_output_shapes
::,-(
&
_output_shapes
:: .

_output_shapes
:: /

_output_shapes
:: 0

_output_shapes
::,1(
&
_output_shapes
: : 2

_output_shapes
: : 3

_output_shapes
: : 4

_output_shapes
: :,5(
&
_output_shapes
: @: 6

_output_shapes
:@: 7

_output_shapes
:@: 8

_output_shapes
:@:-9)
'
_output_shapes
:@А:!:

_output_shapes	
:А:!;

_output_shapes	
:А:!<

_output_shapes	
:А:.=*
(
_output_shapes
:јј:!>

_output_shapes	
:ј:!?

_output_shapes	
:ј:!@

_output_shapes	
:ј:-A)
'
_output_shapes
:а`: B

_output_shapes
:`: C

_output_shapes
:`: D

_output_shapes
:`:,E(
&
_output_shapes
:p0: F

_output_shapes
:0: G

_output_shapes
:0: H

_output_shapes
:0:,I(
&
_output_shapes
:0: J

_output_shapes
::,K(
&
_output_shapes
:: L

_output_shapes
:: M

_output_shapes
:: N

_output_shapes
::,O(
&
_output_shapes
: : P

_output_shapes
: : Q

_output_shapes
: : R

_output_shapes
: :,S(
&
_output_shapes
: @: T

_output_shapes
:@: U

_output_shapes
:@: V

_output_shapes
:@:-W)
'
_output_shapes
:@А:!X

_output_shapes	
:А:!Y

_output_shapes	
:А:!Z

_output_shapes	
:А:.[*
(
_output_shapes
:јј:!\

_output_shapes	
:ј:!]

_output_shapes	
:ј:!^

_output_shapes	
:ј:-_)
'
_output_shapes
:а`: `

_output_shapes
:`: a

_output_shapes
:`: b

_output_shapes
:`:,c(
&
_output_shapes
:p0: d

_output_shapes
:0: e

_output_shapes
:0: f

_output_shapes
:0:,g(
&
_output_shapes
:0: h

_output_shapes
::i

_output_shapes
: 
Ћ
Ќ
2__inference_batch_normalization_layer_call_fn_4103

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identityИҐStatefulPartitionedCallЙ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:€€€€€€€€€АА*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *V
fQRO
M__inference_batch_normalization_layer_call_and_return_conditional_losses_1722y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:€€€€€€€€€АА`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:€€€€€€€€€АА: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:€€€€€€€€€АА
 
_user_specified_nameinputs
и
¬
O__inference_batch_normalization_4_layer_call_and_return_conditional_losses_4844

inputs&
readvariableop_resource:	ј(
readvariableop_1_resource:	ј7
(fusedbatchnormv3_readvariableop_resource:	ј9
*fusedbatchnormv3_readvariableop_1_resource:	ј
identityИҐAssignNewValueҐAssignNewValue_1ҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:ј*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:ј*
dtype0Е
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:ј*
dtype0Й
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:ј*
dtype0џ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,€€€€€€€€€€€€€€€€€€€€€€€€€€€ј:ј:ј:ј:ј:*
epsilon%oГ:*
exponential_avg_factor%
„#<∞
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0Ї
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€ј‘
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,€€€€€€€€€€€€€€€€€€€€€€€€€€€ј: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€ј
 
_user_specified_nameinputs
 
Ъ
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1360

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identityИҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0И
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0»
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@:@:@:@:@:*
epsilon%oГ:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@∞
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs
е
]
A__inference_re_lu_2_layer_call_and_return_conditional_losses_4544

inputs
identityN
ReluReluinputs*
T0*/
_output_shapes
:€€€€€€€€€  @b
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:€€€€€€€€€  @"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:€€€€€€€€€  @:W S
/
_output_shapes
:€€€€€€€€€  @
 
_user_specified_nameinputs
 
Ъ
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_4307

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identityИҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0И
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0»
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ : : : : :*
epsilon%oГ:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ ∞
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ : : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 
 
_user_specified_nameinputs
Џ
Ю
O__inference_batch_normalization_4_layer_call_and_return_conditional_losses_1512

inputs&
readvariableop_resource:	ј(
readvariableop_1_resource:	ј7
(fusedbatchnormv3_readvariableop_resource:	ј9
*fusedbatchnormv3_readvariableop_1_resource:	ј
identityИҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:ј*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:ј*
dtype0Е
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:ј*
dtype0Й
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:ј*
dtype0Ќ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,€€€€€€€€€€€€€€€€€€€€€€€€€€€ј:ј:ј:ј:ј:*
epsilon%oГ:*
is_training( ~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€ј∞
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,€€€€€€€€€€€€€€€€€€€€€€€€€€€ј: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€ј
 
_user_specified_nameinputs
Ў
Њ
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_4325

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identityИҐAssignNewValueҐAssignNewValue_1ҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0И
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0÷
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ : : : : :*
epsilon%oГ:*
exponential_avg_factor%
„#<∞
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0Ї
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ ‘
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ : : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 
 
_user_specified_nameinputs
Ќ
”
4__inference_batch_normalization_3_layer_call_fn_4635

inputs
unknown:	А
	unknown_0:	А
	unknown_1:	А
	unknown_2:	А
identityИҐStatefulPartitionedCallИ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *X
fSRQ
O__inference_batch_normalization_3_layer_call_and_return_conditional_losses_2450x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:€€€€€€€€€А`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:€€€€€€€€€А: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
й
]
A__inference_re_lu_4_layer_call_and_return_conditional_losses_1965

inputs
identityO
ReluReluinputs*
T0*0
_output_shapes
:€€€€€€€€€јc
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:€€€€€€€€€ј"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:€€€€€€€€€ј:X T
0
_output_shapes
:€€€€€€€€€ј
 
_user_specified_nameinputs
Ј
J
.__inference_max_pooling2d_2_layer_call_fn_4549

inputs
identityЏ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *R
fMRK
I__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_1411Г
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:r n
J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
р
Я
'__inference_conv2d_4_layer_call_fn_4746

inputs#
unknown:јј
	unknown_0:	ј
identityИҐStatefulPartitionedCallг
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€ј*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_conv2d_4_layer_call_and_return_conditional_losses_1927x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:€€€€€€€€€ј`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :€€€€€€€€€ј: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:€€€€€€€€€ј
 
_user_specified_nameinputs
 
J
.__inference_max_pooling2d_1_layer_call_fn_4381

inputs
identityњ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€   * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *R
fMRK
I__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_1799h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:€€€€€€€€€   "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:€€€€€€€€€@@ :W S
/
_output_shapes
:€€€€€€€€€@@ 
 
_user_specified_nameinputs
Ц
Љ
M__inference_batch_normalization_layer_call_and_return_conditional_losses_2645

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identityИҐAssignNewValueҐAssignNewValue_1ҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0И
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0∆
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:€€€€€€€€€АА:::::*
epsilon%oГ:*
exponential_avg_factor%
„#<∞
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0Ї
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0m
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*1
_output_shapes
:€€€€€€€€€АА‘
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:€€€€€€€€€АА: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Y U
1
_output_shapes
:€€€€€€€€€АА
 
_user_specified_nameinputs
В
Ъ
O__inference_batch_normalization_6_layer_call_and_return_conditional_losses_2058

inputs%
readvariableop_resource:0'
readvariableop_1_resource:06
(fusedbatchnormv3_readvariableop_resource:08
*fusedbatchnormv3_readvariableop_1_resource:0
identityИҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:0*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:0*
dtype0Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype0И
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype0ґ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€@@0:0:0:0:0:*
epsilon%oГ:*
is_training( k
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:€€€€€€€€€@@0∞
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:€€€€€€€€€@@0: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:€€€€€€€€€@@0
 
_user_specified_nameinputs
Ў
Њ
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1391

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identityИҐAssignNewValueҐAssignNewValue_1ҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0И
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0÷
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@:@:@:@:@:*
epsilon%oГ:*
exponential_avg_factor%
„#<∞
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0Ї
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@‘
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs
’С
Д,
E__inference_Mobile_Unet_layer_call_and_return_conditional_losses_4045

inputs?
%conv2d_conv2d_readvariableop_resource:4
&conv2d_biasadd_readvariableop_resource:9
+batch_normalization_readvariableop_resource:;
-batch_normalization_readvariableop_1_resource:J
<batch_normalization_fusedbatchnormv3_readvariableop_resource:L
>batch_normalization_fusedbatchnormv3_readvariableop_1_resource:A
'conv2d_1_conv2d_readvariableop_resource: 6
(conv2d_1_biasadd_readvariableop_resource: ;
-batch_normalization_1_readvariableop_resource: =
/batch_normalization_1_readvariableop_1_resource: L
>batch_normalization_1_fusedbatchnormv3_readvariableop_resource: N
@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource: A
'conv2d_2_conv2d_readvariableop_resource: @6
(conv2d_2_biasadd_readvariableop_resource:@;
-batch_normalization_2_readvariableop_resource:@=
/batch_normalization_2_readvariableop_1_resource:@L
>batch_normalization_2_fusedbatchnormv3_readvariableop_resource:@N
@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource:@B
'conv2d_3_conv2d_readvariableop_resource:@А7
(conv2d_3_biasadd_readvariableop_resource:	А<
-batch_normalization_3_readvariableop_resource:	А>
/batch_normalization_3_readvariableop_1_resource:	АM
>batch_normalization_3_fusedbatchnormv3_readvariableop_resource:	АO
@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource:	А
tf_image_resize_resize_sizeC
'conv2d_4_conv2d_readvariableop_resource:јј7
(conv2d_4_biasadd_readvariableop_resource:	ј<
-batch_normalization_4_readvariableop_resource:	ј>
/batch_normalization_4_readvariableop_1_resource:	јM
>batch_normalization_4_fusedbatchnormv3_readvariableop_resource:	јO
@batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource:	ј!
tf_image_resize_1_resize_sizeB
'conv2d_5_conv2d_readvariableop_resource:а`6
(conv2d_5_biasadd_readvariableop_resource:`;
-batch_normalization_5_readvariableop_resource:`=
/batch_normalization_5_readvariableop_1_resource:`L
>batch_normalization_5_fusedbatchnormv3_readvariableop_resource:`N
@batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource:`!
tf_image_resize_2_resize_sizeA
'conv2d_6_conv2d_readvariableop_resource:p06
(conv2d_6_biasadd_readvariableop_resource:0;
-batch_normalization_6_readvariableop_resource:0=
/batch_normalization_6_readvariableop_1_resource:0L
>batch_normalization_6_fusedbatchnormv3_readvariableop_resource:0N
@batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource:0!
tf_image_resize_3_resize_sizeA
'conv2d_7_conv2d_readvariableop_resource:06
(conv2d_7_biasadd_readvariableop_resource:
identityИҐ"batch_normalization/AssignNewValueҐ$batch_normalization/AssignNewValue_1Ґ3batch_normalization/FusedBatchNormV3/ReadVariableOpҐ5batch_normalization/FusedBatchNormV3/ReadVariableOp_1Ґ"batch_normalization/ReadVariableOpҐ$batch_normalization/ReadVariableOp_1Ґ$batch_normalization_1/AssignNewValueҐ&batch_normalization_1/AssignNewValue_1Ґ5batch_normalization_1/FusedBatchNormV3/ReadVariableOpҐ7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1Ґ$batch_normalization_1/ReadVariableOpҐ&batch_normalization_1/ReadVariableOp_1Ґ$batch_normalization_2/AssignNewValueҐ&batch_normalization_2/AssignNewValue_1Ґ5batch_normalization_2/FusedBatchNormV3/ReadVariableOpҐ7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1Ґ$batch_normalization_2/ReadVariableOpҐ&batch_normalization_2/ReadVariableOp_1Ґ$batch_normalization_3/AssignNewValueҐ&batch_normalization_3/AssignNewValue_1Ґ5batch_normalization_3/FusedBatchNormV3/ReadVariableOpҐ7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1Ґ$batch_normalization_3/ReadVariableOpҐ&batch_normalization_3/ReadVariableOp_1Ґ$batch_normalization_4/AssignNewValueҐ&batch_normalization_4/AssignNewValue_1Ґ5batch_normalization_4/FusedBatchNormV3/ReadVariableOpҐ7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1Ґ$batch_normalization_4/ReadVariableOpҐ&batch_normalization_4/ReadVariableOp_1Ґ$batch_normalization_5/AssignNewValueҐ&batch_normalization_5/AssignNewValue_1Ґ5batch_normalization_5/FusedBatchNormV3/ReadVariableOpҐ7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1Ґ$batch_normalization_5/ReadVariableOpҐ&batch_normalization_5/ReadVariableOp_1Ґ$batch_normalization_6/AssignNewValueҐ&batch_normalization_6/AssignNewValue_1Ґ5batch_normalization_6/FusedBatchNormV3/ReadVariableOpҐ7batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1Ґ$batch_normalization_6/ReadVariableOpҐ&batch_normalization_6/ReadVariableOp_1Ґconv2d/BiasAdd/ReadVariableOpҐconv2d/Conv2D/ReadVariableOpҐconv2d_1/BiasAdd/ReadVariableOpҐconv2d_1/Conv2D/ReadVariableOpҐconv2d_2/BiasAdd/ReadVariableOpҐconv2d_2/Conv2D/ReadVariableOpҐconv2d_3/BiasAdd/ReadVariableOpҐconv2d_3/Conv2D/ReadVariableOpҐconv2d_4/BiasAdd/ReadVariableOpҐconv2d_4/Conv2D/ReadVariableOpҐconv2d_5/BiasAdd/ReadVariableOpҐconv2d_5/Conv2D/ReadVariableOpҐconv2d_6/BiasAdd/ReadVariableOpҐconv2d_6/Conv2D/ReadVariableOpҐconv2d_7/BiasAdd/ReadVariableOpҐconv2d_7/Conv2D/ReadVariableOpК
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0©
conv2d/Conv2DConv2Dinputs$conv2d/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:€€€€€€€€€АА*
paddingSAME*
strides
А
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ф
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:€€€€€€€€€ААК
"batch_normalization/ReadVariableOpReadVariableOp+batch_normalization_readvariableop_resource*
_output_shapes
:*
dtype0О
$batch_normalization/ReadVariableOp_1ReadVariableOp-batch_normalization_readvariableop_1_resource*
_output_shapes
:*
dtype0ђ
3batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOp<batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0∞
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp>batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0ї
$batch_normalization/FusedBatchNormV3FusedBatchNormV3conv2d/BiasAdd:output:0*batch_normalization/ReadVariableOp:value:0,batch_normalization/ReadVariableOp_1:value:0;batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0=batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:€€€€€€€€€АА:::::*
epsilon%oГ:*
exponential_avg_factor%
„#<А
"batch_normalization/AssignNewValueAssignVariableOp<batch_normalization_fusedbatchnormv3_readvariableop_resource1batch_normalization/FusedBatchNormV3:batch_mean:04^batch_normalization/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0К
$batch_normalization/AssignNewValue_1AssignVariableOp>batch_normalization_fusedbatchnormv3_readvariableop_1_resource5batch_normalization/FusedBatchNormV3:batch_variance:06^batch_normalization/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0x

re_lu/ReluRelu(batch_normalization/FusedBatchNormV3:y:0*
T0*1
_output_shapes
:€€€€€€€€€АА¶
max_pooling2d/MaxPoolMaxPoolre_lu/Relu:activations:0*/
_output_shapes
:€€€€€€€€€@@*
ksize
*
paddingSAME*
strides
О
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0√
conv2d_1/Conv2DConv2Dmax_pooling2d/MaxPool:output:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€@@ *
paddingSAME*
strides
Д
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0Ш
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€@@ О
$batch_normalization_1/ReadVariableOpReadVariableOp-batch_normalization_1_readvariableop_resource*
_output_shapes
: *
dtype0Т
&batch_normalization_1/ReadVariableOp_1ReadVariableOp/batch_normalization_1_readvariableop_1_resource*
_output_shapes
: *
dtype0∞
5batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0і
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0≈
&batch_normalization_1/FusedBatchNormV3FusedBatchNormV3conv2d_1/BiasAdd:output:0,batch_normalization_1/ReadVariableOp:value:0.batch_normalization_1/ReadVariableOp_1:value:0=batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€@@ : : : : :*
epsilon%oГ:*
exponential_avg_factor%
„#<И
$batch_normalization_1/AssignNewValueAssignVariableOp>batch_normalization_1_fusedbatchnormv3_readvariableop_resource3batch_normalization_1/FusedBatchNormV3:batch_mean:06^batch_normalization_1/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0Т
&batch_normalization_1/AssignNewValue_1AssignVariableOp@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_1/FusedBatchNormV3:batch_variance:08^batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0z
re_lu_1/ReluRelu*batch_normalization_1/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:€€€€€€€€€@@ ™
max_pooling2d_1/MaxPoolMaxPoolre_lu_1/Relu:activations:0*/
_output_shapes
:€€€€€€€€€   *
ksize
*
paddingSAME*
strides
О
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0≈
conv2d_2/Conv2DConv2D max_pooling2d_1/MaxPool:output:0&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€  @*
paddingSAME*
strides
Д
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Ш
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€  @О
$batch_normalization_2/ReadVariableOpReadVariableOp-batch_normalization_2_readvariableop_resource*
_output_shapes
:@*
dtype0Т
&batch_normalization_2/ReadVariableOp_1ReadVariableOp/batch_normalization_2_readvariableop_1_resource*
_output_shapes
:@*
dtype0∞
5batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0і
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0≈
&batch_normalization_2/FusedBatchNormV3FusedBatchNormV3conv2d_2/BiasAdd:output:0,batch_normalization_2/ReadVariableOp:value:0.batch_normalization_2/ReadVariableOp_1:value:0=batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€  @:@:@:@:@:*
epsilon%oГ:*
exponential_avg_factor%
„#<И
$batch_normalization_2/AssignNewValueAssignVariableOp>batch_normalization_2_fusedbatchnormv3_readvariableop_resource3batch_normalization_2/FusedBatchNormV3:batch_mean:06^batch_normalization_2/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0Т
&batch_normalization_2/AssignNewValue_1AssignVariableOp@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_2/FusedBatchNormV3:batch_variance:08^batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0z
re_lu_2/ReluRelu*batch_normalization_2/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:€€€€€€€€€  @™
max_pooling2d_2/MaxPoolMaxPoolre_lu_2/Relu:activations:0*/
_output_shapes
:€€€€€€€€€@*
ksize
*
paddingSAME*
strides
П
conv2d_3/Conv2D/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*'
_output_shapes
:@А*
dtype0∆
conv2d_3/Conv2DConv2D max_pooling2d_2/MaxPool:output:0&conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А*
paddingSAME*
strides
Е
conv2d_3/BiasAdd/ReadVariableOpReadVariableOp(conv2d_3_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0Щ
conv2d_3/BiasAddBiasAddconv2d_3/Conv2D:output:0'conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€АП
$batch_normalization_3/ReadVariableOpReadVariableOp-batch_normalization_3_readvariableop_resource*
_output_shapes	
:А*
dtype0У
&batch_normalization_3/ReadVariableOp_1ReadVariableOp/batch_normalization_3_readvariableop_1_resource*
_output_shapes	
:А*
dtype0±
5batch_normalization_3/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype0µ
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype0 
&batch_normalization_3/FusedBatchNormV3FusedBatchNormV3conv2d_3/BiasAdd:output:0,batch_normalization_3/ReadVariableOp:value:0.batch_normalization_3/ReadVariableOp_1:value:0=batch_normalization_3/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:€€€€€€€€€А:А:А:А:А:*
epsilon%oГ:*
exponential_avg_factor%
„#<И
$batch_normalization_3/AssignNewValueAssignVariableOp>batch_normalization_3_fusedbatchnormv3_readvariableop_resource3batch_normalization_3/FusedBatchNormV3:batch_mean:06^batch_normalization_3/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0Т
&batch_normalization_3/AssignNewValue_1AssignVariableOp@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_3/FusedBatchNormV3:batch_variance:08^batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0{
re_lu_3/ReluRelu*batch_normalization_3/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:€€€€€€€€€АЂ
max_pooling2d_3/MaxPoolMaxPoolre_lu_3/Relu:activations:0*0
_output_shapes
:€€€€€€€€€А*
ksize
*
paddingSAME*
strides
ў
,tf.image.resize/resize/ResizeNearestNeighborResizeNearestNeighbor max_pooling2d_3/MaxPool:output:0tf_image_resize_resize_size*
T0*0
_output_shapes
:€€€€€€€€€А*
half_pixel_centers(W
tf.concat/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :б
tf.concat/concatConcatV2=tf.image.resize/resize/ResizeNearestNeighbor:resized_images:0 max_pooling2d_2/MaxPool:output:0tf.concat/concat/axis:output:0*
N*
T0*0
_output_shapes
:€€€€€€€€€јР
conv2d_4/Conv2D/ReadVariableOpReadVariableOp'conv2d_4_conv2d_readvariableop_resource*(
_output_shapes
:јј*
dtype0њ
conv2d_4/Conv2DConv2Dtf.concat/concat:output:0&conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€ј*
paddingSAME*
strides
Е
conv2d_4/BiasAdd/ReadVariableOpReadVariableOp(conv2d_4_biasadd_readvariableop_resource*
_output_shapes	
:ј*
dtype0Щ
conv2d_4/BiasAddBiasAddconv2d_4/Conv2D:output:0'conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€јП
$batch_normalization_4/ReadVariableOpReadVariableOp-batch_normalization_4_readvariableop_resource*
_output_shapes	
:ј*
dtype0У
&batch_normalization_4/ReadVariableOp_1ReadVariableOp/batch_normalization_4_readvariableop_1_resource*
_output_shapes	
:ј*
dtype0±
5batch_normalization_4/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_4_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:ј*
dtype0µ
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:ј*
dtype0 
&batch_normalization_4/FusedBatchNormV3FusedBatchNormV3conv2d_4/BiasAdd:output:0,batch_normalization_4/ReadVariableOp:value:0.batch_normalization_4/ReadVariableOp_1:value:0=batch_normalization_4/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:€€€€€€€€€ј:ј:ј:ј:ј:*
epsilon%oГ:*
exponential_avg_factor%
„#<И
$batch_normalization_4/AssignNewValueAssignVariableOp>batch_normalization_4_fusedbatchnormv3_readvariableop_resource3batch_normalization_4/FusedBatchNormV3:batch_mean:06^batch_normalization_4/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0Т
&batch_normalization_4/AssignNewValue_1AssignVariableOp@batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_4/FusedBatchNormV3:batch_variance:08^batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0{
re_lu_4/ReluRelu*batch_normalization_4/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:€€€€€€€€€ј„
.tf.image.resize_1/resize/ResizeNearestNeighborResizeNearestNeighborre_lu_4/Relu:activations:0tf_image_resize_1_resize_size*
T0*0
_output_shapes
:€€€€€€€€€  ј*
half_pixel_centers(Y
tf.concat_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :з
tf.concat_1/concatConcatV2?tf.image.resize_1/resize/ResizeNearestNeighbor:resized_images:0 max_pooling2d_1/MaxPool:output:0 tf.concat_1/concat/axis:output:0*
N*
T0*0
_output_shapes
:€€€€€€€€€  аП
conv2d_5/Conv2D/ReadVariableOpReadVariableOp'conv2d_5_conv2d_readvariableop_resource*'
_output_shapes
:а`*
dtype0ј
conv2d_5/Conv2DConv2Dtf.concat_1/concat:output:0&conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€  `*
paddingSAME*
strides
Д
conv2d_5/BiasAdd/ReadVariableOpReadVariableOp(conv2d_5_biasadd_readvariableop_resource*
_output_shapes
:`*
dtype0Ш
conv2d_5/BiasAddBiasAddconv2d_5/Conv2D:output:0'conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€  `О
$batch_normalization_5/ReadVariableOpReadVariableOp-batch_normalization_5_readvariableop_resource*
_output_shapes
:`*
dtype0Т
&batch_normalization_5/ReadVariableOp_1ReadVariableOp/batch_normalization_5_readvariableop_1_resource*
_output_shapes
:`*
dtype0∞
5batch_normalization_5/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_5_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:`*
dtype0і
7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:`*
dtype0≈
&batch_normalization_5/FusedBatchNormV3FusedBatchNormV3conv2d_5/BiasAdd:output:0,batch_normalization_5/ReadVariableOp:value:0.batch_normalization_5/ReadVariableOp_1:value:0=batch_normalization_5/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€  `:`:`:`:`:*
epsilon%oГ:*
exponential_avg_factor%
„#<И
$batch_normalization_5/AssignNewValueAssignVariableOp>batch_normalization_5_fusedbatchnormv3_readvariableop_resource3batch_normalization_5/FusedBatchNormV3:batch_mean:06^batch_normalization_5/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0Т
&batch_normalization_5/AssignNewValue_1AssignVariableOp@batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_5/FusedBatchNormV3:batch_variance:08^batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0z
re_lu_5/ReluRelu*batch_normalization_5/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:€€€€€€€€€  `÷
.tf.image.resize_2/resize/ResizeNearestNeighborResizeNearestNeighborre_lu_5/Relu:activations:0tf_image_resize_2_resize_size*
T0*/
_output_shapes
:€€€€€€€€€@@`*
half_pixel_centers(Y
tf.concat_2/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :д
tf.concat_2/concatConcatV2?tf.image.resize_2/resize/ResizeNearestNeighbor:resized_images:0max_pooling2d/MaxPool:output:0 tf.concat_2/concat/axis:output:0*
N*
T0*/
_output_shapes
:€€€€€€€€€@@pО
conv2d_6/Conv2D/ReadVariableOpReadVariableOp'conv2d_6_conv2d_readvariableop_resource*&
_output_shapes
:p0*
dtype0ј
conv2d_6/Conv2DConv2Dtf.concat_2/concat:output:0&conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€@@0*
paddingSAME*
strides
Д
conv2d_6/BiasAdd/ReadVariableOpReadVariableOp(conv2d_6_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype0Ш
conv2d_6/BiasAddBiasAddconv2d_6/Conv2D:output:0'conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€@@0О
$batch_normalization_6/ReadVariableOpReadVariableOp-batch_normalization_6_readvariableop_resource*
_output_shapes
:0*
dtype0Т
&batch_normalization_6/ReadVariableOp_1ReadVariableOp/batch_normalization_6_readvariableop_1_resource*
_output_shapes
:0*
dtype0∞
5batch_normalization_6/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_6_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype0і
7batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype0≈
&batch_normalization_6/FusedBatchNormV3FusedBatchNormV3conv2d_6/BiasAdd:output:0,batch_normalization_6/ReadVariableOp:value:0.batch_normalization_6/ReadVariableOp_1:value:0=batch_normalization_6/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€@@0:0:0:0:0:*
epsilon%oГ:*
exponential_avg_factor%
„#<И
$batch_normalization_6/AssignNewValueAssignVariableOp>batch_normalization_6_fusedbatchnormv3_readvariableop_resource3batch_normalization_6/FusedBatchNormV3:batch_mean:06^batch_normalization_6/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0Т
&batch_normalization_6/AssignNewValue_1AssignVariableOp@batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_6/FusedBatchNormV3:batch_variance:08^batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0z
re_lu_6/ReluRelu*batch_normalization_6/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:€€€€€€€€€@@0Ў
.tf.image.resize_3/resize/ResizeNearestNeighborResizeNearestNeighborre_lu_6/Relu:activations:0tf_image_resize_3_resize_size*
T0*1
_output_shapes
:€€€€€€€€€АА0*
half_pixel_centers(О
conv2d_7/Conv2D/ReadVariableOpReadVariableOp'conv2d_7_conv2d_readvariableop_resource*&
_output_shapes
:0*
dtype0ж
conv2d_7/Conv2DConv2D?tf.image.resize_3/resize/ResizeNearestNeighbor:resized_images:0&conv2d_7/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:€€€€€€€€€АА*
paddingSAME*
strides
Д
conv2d_7/BiasAdd/ReadVariableOpReadVariableOp(conv2d_7_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ъ
conv2d_7/BiasAddBiasAddconv2d_7/Conv2D:output:0'conv2d_7/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:€€€€€€€€€ААr
conv2d_7/SigmoidSigmoidconv2d_7/BiasAdd:output:0*
T0*1
_output_shapes
:€€€€€€€€€ААQ
reshape/ShapeShapeconv2d_7/Sigmoid:y:0*
T0*
_output_shapes
:e
reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: g
reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:g
reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:щ
reshape/strided_sliceStridedSlicereshape/Shape:output:0$reshape/strided_slice/stack:output:0&reshape/strided_slice/stack_1:output:0&reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskY
reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :Z
reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value
B :АZ
reshape/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value
B :А—
reshape/Reshape/shapePackreshape/strided_slice:output:0 reshape/Reshape/shape/1:output:0 reshape/Reshape/shape/2:output:0 reshape/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:М
reshape/ReshapeReshapeconv2d_7/Sigmoid:y:0reshape/Reshape/shape:output:0*
T0*1
_output_shapes
:€€€€€€€€€ААq
IdentityIdentityreshape/Reshape:output:0^NoOp*
T0*1
_output_shapes
:€€€€€€€€€ААћ
NoOpNoOp#^batch_normalization/AssignNewValue%^batch_normalization/AssignNewValue_14^batch_normalization/FusedBatchNormV3/ReadVariableOp6^batch_normalization/FusedBatchNormV3/ReadVariableOp_1#^batch_normalization/ReadVariableOp%^batch_normalization/ReadVariableOp_1%^batch_normalization_1/AssignNewValue'^batch_normalization_1/AssignNewValue_16^batch_normalization_1/FusedBatchNormV3/ReadVariableOp8^batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_1/ReadVariableOp'^batch_normalization_1/ReadVariableOp_1%^batch_normalization_2/AssignNewValue'^batch_normalization_2/AssignNewValue_16^batch_normalization_2/FusedBatchNormV3/ReadVariableOp8^batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_2/ReadVariableOp'^batch_normalization_2/ReadVariableOp_1%^batch_normalization_3/AssignNewValue'^batch_normalization_3/AssignNewValue_16^batch_normalization_3/FusedBatchNormV3/ReadVariableOp8^batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_3/ReadVariableOp'^batch_normalization_3/ReadVariableOp_1%^batch_normalization_4/AssignNewValue'^batch_normalization_4/AssignNewValue_16^batch_normalization_4/FusedBatchNormV3/ReadVariableOp8^batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_4/ReadVariableOp'^batch_normalization_4/ReadVariableOp_1%^batch_normalization_5/AssignNewValue'^batch_normalization_5/AssignNewValue_16^batch_normalization_5/FusedBatchNormV3/ReadVariableOp8^batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_5/ReadVariableOp'^batch_normalization_5/ReadVariableOp_1%^batch_normalization_6/AssignNewValue'^batch_normalization_6/AssignNewValue_16^batch_normalization_6/FusedBatchNormV3/ReadVariableOp8^batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_6/ReadVariableOp'^batch_normalization_6/ReadVariableOp_1^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp ^conv2d_3/BiasAdd/ReadVariableOp^conv2d_3/Conv2D/ReadVariableOp ^conv2d_4/BiasAdd/ReadVariableOp^conv2d_4/Conv2D/ReadVariableOp ^conv2d_5/BiasAdd/ReadVariableOp^conv2d_5/Conv2D/ReadVariableOp ^conv2d_6/BiasAdd/ReadVariableOp^conv2d_6/Conv2D/ReadVariableOp ^conv2d_7/BiasAdd/ReadVariableOp^conv2d_7/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Ґ
_input_shapesР
Н:€€€€€€€€€АА: : : : : : : : : : : : : : : : : : : : : : : : :: : : : : : :: : : : : : :: : : : : : :: : 2H
"batch_normalization/AssignNewValue"batch_normalization/AssignNewValue2L
$batch_normalization/AssignNewValue_1$batch_normalization/AssignNewValue_12j
3batch_normalization/FusedBatchNormV3/ReadVariableOp3batch_normalization/FusedBatchNormV3/ReadVariableOp2n
5batch_normalization/FusedBatchNormV3/ReadVariableOp_15batch_normalization/FusedBatchNormV3/ReadVariableOp_12H
"batch_normalization/ReadVariableOp"batch_normalization/ReadVariableOp2L
$batch_normalization/ReadVariableOp_1$batch_normalization/ReadVariableOp_12L
$batch_normalization_1/AssignNewValue$batch_normalization_1/AssignNewValue2P
&batch_normalization_1/AssignNewValue_1&batch_normalization_1/AssignNewValue_12n
5batch_normalization_1/FusedBatchNormV3/ReadVariableOp5batch_normalization_1/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_17batch_normalization_1/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_1/ReadVariableOp$batch_normalization_1/ReadVariableOp2P
&batch_normalization_1/ReadVariableOp_1&batch_normalization_1/ReadVariableOp_12L
$batch_normalization_2/AssignNewValue$batch_normalization_2/AssignNewValue2P
&batch_normalization_2/AssignNewValue_1&batch_normalization_2/AssignNewValue_12n
5batch_normalization_2/FusedBatchNormV3/ReadVariableOp5batch_normalization_2/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_17batch_normalization_2/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_2/ReadVariableOp$batch_normalization_2/ReadVariableOp2P
&batch_normalization_2/ReadVariableOp_1&batch_normalization_2/ReadVariableOp_12L
$batch_normalization_3/AssignNewValue$batch_normalization_3/AssignNewValue2P
&batch_normalization_3/AssignNewValue_1&batch_normalization_3/AssignNewValue_12n
5batch_normalization_3/FusedBatchNormV3/ReadVariableOp5batch_normalization_3/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_17batch_normalization_3/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_3/ReadVariableOp$batch_normalization_3/ReadVariableOp2P
&batch_normalization_3/ReadVariableOp_1&batch_normalization_3/ReadVariableOp_12L
$batch_normalization_4/AssignNewValue$batch_normalization_4/AssignNewValue2P
&batch_normalization_4/AssignNewValue_1&batch_normalization_4/AssignNewValue_12n
5batch_normalization_4/FusedBatchNormV3/ReadVariableOp5batch_normalization_4/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_17batch_normalization_4/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_4/ReadVariableOp$batch_normalization_4/ReadVariableOp2P
&batch_normalization_4/ReadVariableOp_1&batch_normalization_4/ReadVariableOp_12L
$batch_normalization_5/AssignNewValue$batch_normalization_5/AssignNewValue2P
&batch_normalization_5/AssignNewValue_1&batch_normalization_5/AssignNewValue_12n
5batch_normalization_5/FusedBatchNormV3/ReadVariableOp5batch_normalization_5/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_17batch_normalization_5/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_5/ReadVariableOp$batch_normalization_5/ReadVariableOp2P
&batch_normalization_5/ReadVariableOp_1&batch_normalization_5/ReadVariableOp_12L
$batch_normalization_6/AssignNewValue$batch_normalization_6/AssignNewValue2P
&batch_normalization_6/AssignNewValue_1&batch_normalization_6/AssignNewValue_12n
5batch_normalization_6/FusedBatchNormV3/ReadVariableOp5batch_normalization_6/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_6/FusedBatchNormV3/ReadVariableOp_17batch_normalization_6/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_6/ReadVariableOp$batch_normalization_6/ReadVariableOp2P
&batch_normalization_6/ReadVariableOp_1&batch_normalization_6/ReadVariableOp_12>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp2B
conv2d_2/BiasAdd/ReadVariableOpconv2d_2/BiasAdd/ReadVariableOp2@
conv2d_2/Conv2D/ReadVariableOpconv2d_2/Conv2D/ReadVariableOp2B
conv2d_3/BiasAdd/ReadVariableOpconv2d_3/BiasAdd/ReadVariableOp2@
conv2d_3/Conv2D/ReadVariableOpconv2d_3/Conv2D/ReadVariableOp2B
conv2d_4/BiasAdd/ReadVariableOpconv2d_4/BiasAdd/ReadVariableOp2@
conv2d_4/Conv2D/ReadVariableOpconv2d_4/Conv2D/ReadVariableOp2B
conv2d_5/BiasAdd/ReadVariableOpconv2d_5/BiasAdd/ReadVariableOp2@
conv2d_5/Conv2D/ReadVariableOpconv2d_5/Conv2D/ReadVariableOp2B
conv2d_6/BiasAdd/ReadVariableOpconv2d_6/BiasAdd/ReadVariableOp2@
conv2d_6/Conv2D/ReadVariableOpconv2d_6/Conv2D/ReadVariableOp2B
conv2d_7/BiasAdd/ReadVariableOpconv2d_7/BiasAdd/ReadVariableOp2@
conv2d_7/Conv2D/ReadVariableOpconv2d_7/Conv2D/ReadVariableOp:Y U
1
_output_shapes
:€€€€€€€€€АА
 
_user_specified_nameinputs: 

_output_shapes
::  

_output_shapes
:: '

_output_shapes
:: .

_output_shapes
:
ћО
Э
E__inference_Mobile_Unet_layer_call_and_return_conditional_losses_3235
input_1%
conv2d_3105:
conv2d_3107:&
batch_normalization_3110:&
batch_normalization_3112:&
batch_normalization_3114:&
batch_normalization_3116:'
conv2d_1_3121: 
conv2d_1_3123: (
batch_normalization_1_3126: (
batch_normalization_1_3128: (
batch_normalization_1_3130: (
batch_normalization_1_3132: '
conv2d_2_3137: @
conv2d_2_3139:@(
batch_normalization_2_3142:@(
batch_normalization_2_3144:@(
batch_normalization_2_3146:@(
batch_normalization_2_3148:@(
conv2d_3_3153:@А
conv2d_3_3155:	А)
batch_normalization_3_3158:	А)
batch_normalization_3_3160:	А)
batch_normalization_3_3162:	А)
batch_normalization_3_3164:	А
tf_image_resize_resize_size)
conv2d_4_3173:јј
conv2d_4_3175:	ј)
batch_normalization_4_3178:	ј)
batch_normalization_4_3180:	ј)
batch_normalization_4_3182:	ј)
batch_normalization_4_3184:	ј!
tf_image_resize_1_resize_size(
conv2d_5_3192:а`
conv2d_5_3194:`(
batch_normalization_5_3197:`(
batch_normalization_5_3199:`(
batch_normalization_5_3201:`(
batch_normalization_5_3203:`!
tf_image_resize_2_resize_size'
conv2d_6_3211:p0
conv2d_6_3213:0(
batch_normalization_6_3216:0(
batch_normalization_6_3218:0(
batch_normalization_6_3220:0(
batch_normalization_6_3222:0!
tf_image_resize_3_resize_size'
conv2d_7_3228:0
conv2d_7_3230:
identityИҐ+batch_normalization/StatefulPartitionedCallҐ-batch_normalization_1/StatefulPartitionedCallҐ-batch_normalization_2/StatefulPartitionedCallҐ-batch_normalization_3/StatefulPartitionedCallҐ-batch_normalization_4/StatefulPartitionedCallҐ-batch_normalization_5/StatefulPartitionedCallҐ-batch_normalization_6/StatefulPartitionedCallҐconv2d/StatefulPartitionedCallҐ conv2d_1/StatefulPartitionedCallҐ conv2d_2/StatefulPartitionedCallҐ conv2d_3/StatefulPartitionedCallҐ conv2d_4/StatefulPartitionedCallҐ conv2d_5/StatefulPartitionedCallҐ conv2d_6/StatefulPartitionedCallҐ conv2d_7/StatefulPartitionedCallр
conv2d/StatefulPartitionedCallStatefulPartitionedCallinput_1conv2d_3105conv2d_3107*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:€€€€€€€€€АА*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *I
fDRB
@__inference_conv2d_layer_call_and_return_conditional_losses_1699ь
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0batch_normalization_3110batch_normalization_3112batch_normalization_3114batch_normalization_3116*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:€€€€€€€€€АА*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *V
fQRO
M__inference_batch_normalization_layer_call_and_return_conditional_losses_1722л
re_lu/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:€€€€€€€€€АА* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *H
fCRA
?__inference_re_lu_layer_call_and_return_conditional_losses_1737г
max_pooling2d/PartitionedCallPartitionedCallre_lu/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€@@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_max_pooling2d_layer_call_and_return_conditional_losses_1743Х
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0conv2d_1_3121conv2d_1_3123*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€@@ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_conv2d_1_layer_call_and_return_conditional_losses_1755И
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0batch_normalization_1_3126batch_normalization_1_3128batch_normalization_1_3130batch_normalization_1_3132*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€@@ *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *X
fSRQ
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1778п
re_lu_1/PartitionedCallPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€@@ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_re_lu_1_layer_call_and_return_conditional_losses_1793й
max_pooling2d_1/PartitionedCallPartitionedCall re_lu_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€   * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *R
fMRK
I__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_1799Ч
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_1/PartitionedCall:output:0conv2d_2_3137conv2d_2_3139*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€  @*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_conv2d_2_layer_call_and_return_conditional_losses_1811И
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0batch_normalization_2_3142batch_normalization_2_3144batch_normalization_2_3146batch_normalization_2_3148*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€  @*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *X
fSRQ
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1834п
re_lu_2/PartitionedCallPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€  @* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_re_lu_2_layer_call_and_return_conditional_losses_1849й
max_pooling2d_2/PartitionedCallPartitionedCall re_lu_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *R
fMRK
I__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_1855Ш
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_2/PartitionedCall:output:0conv2d_3_3153conv2d_3_3155*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_conv2d_3_layer_call_and_return_conditional_losses_1867Й
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0batch_normalization_3_3158batch_normalization_3_3160batch_normalization_3_3162batch_normalization_3_3164*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€А*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *X
fSRQ
O__inference_batch_normalization_3_layer_call_and_return_conditional_losses_1890р
re_lu_3/PartitionedCallPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_re_lu_3_layer_call_and_return_conditional_losses_1905к
max_pooling2d_3/PartitionedCallPartitionedCall re_lu_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *R
fMRK
I__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_1911б
,tf.image.resize/resize/ResizeNearestNeighborResizeNearestNeighbor(max_pooling2d_3/PartitionedCall:output:0tf_image_resize_resize_size*
T0*0
_output_shapes
:€€€€€€€€€А*
half_pixel_centers(W
tf.concat/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :й
tf.concat/concatConcatV2=tf.image.resize/resize/ResizeNearestNeighbor:resized_images:0(max_pooling2d_2/PartitionedCall:output:0tf.concat/concat/axis:output:0*
N*
T0*0
_output_shapes
:€€€€€€€€€јЙ
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCalltf.concat/concat:output:0conv2d_4_3173conv2d_4_3175*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€ј*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_conv2d_4_layer_call_and_return_conditional_losses_1927Й
-batch_normalization_4/StatefulPartitionedCallStatefulPartitionedCall)conv2d_4/StatefulPartitionedCall:output:0batch_normalization_4_3178batch_normalization_4_3180batch_normalization_4_3182batch_normalization_4_3184*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€ј*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *X
fSRQ
O__inference_batch_normalization_4_layer_call_and_return_conditional_losses_1950р
re_lu_4/PartitionedCallPartitionedCall6batch_normalization_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€ј* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_re_lu_4_layer_call_and_return_conditional_losses_1965Ё
.tf.image.resize_1/resize/ResizeNearestNeighborResizeNearestNeighbor re_lu_4/PartitionedCall:output:0tf_image_resize_1_resize_size*
T0*0
_output_shapes
:€€€€€€€€€  ј*
half_pixel_centers(Y
tf.concat_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :п
tf.concat_1/concatConcatV2?tf.image.resize_1/resize/ResizeNearestNeighbor:resized_images:0(max_pooling2d_1/PartitionedCall:output:0 tf.concat_1/concat/axis:output:0*
N*
T0*0
_output_shapes
:€€€€€€€€€  аК
 conv2d_5/StatefulPartitionedCallStatefulPartitionedCalltf.concat_1/concat:output:0conv2d_5_3192conv2d_5_3194*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€  `*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_conv2d_5_layer_call_and_return_conditional_losses_1981И
-batch_normalization_5/StatefulPartitionedCallStatefulPartitionedCall)conv2d_5/StatefulPartitionedCall:output:0batch_normalization_5_3197batch_normalization_5_3199batch_normalization_5_3201batch_normalization_5_3203*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€  `*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *X
fSRQ
O__inference_batch_normalization_5_layer_call_and_return_conditional_losses_2004п
re_lu_5/PartitionedCallPartitionedCall6batch_normalization_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€  `* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_re_lu_5_layer_call_and_return_conditional_losses_2019№
.tf.image.resize_2/resize/ResizeNearestNeighborResizeNearestNeighbor re_lu_5/PartitionedCall:output:0tf_image_resize_2_resize_size*
T0*/
_output_shapes
:€€€€€€€€€@@`*
half_pixel_centers(Y
tf.concat_2/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :м
tf.concat_2/concatConcatV2?tf.image.resize_2/resize/ResizeNearestNeighbor:resized_images:0&max_pooling2d/PartitionedCall:output:0 tf.concat_2/concat/axis:output:0*
N*
T0*/
_output_shapes
:€€€€€€€€€@@pК
 conv2d_6/StatefulPartitionedCallStatefulPartitionedCalltf.concat_2/concat:output:0conv2d_6_3211conv2d_6_3213*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€@@0*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_conv2d_6_layer_call_and_return_conditional_losses_2035И
-batch_normalization_6/StatefulPartitionedCallStatefulPartitionedCall)conv2d_6/StatefulPartitionedCall:output:0batch_normalization_6_3216batch_normalization_6_3218batch_normalization_6_3220batch_normalization_6_3222*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€@@0*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *X
fSRQ
O__inference_batch_normalization_6_layer_call_and_return_conditional_losses_2058п
re_lu_6/PartitionedCallPartitionedCall6batch_normalization_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€@@0* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_re_lu_6_layer_call_and_return_conditional_losses_2073ё
.tf.image.resize_3/resize/ResizeNearestNeighborResizeNearestNeighbor re_lu_6/PartitionedCall:output:0tf_image_resize_3_resize_size*
T0*1
_output_shapes
:€€€€€€€€€АА0*
half_pixel_centers(∞
 conv2d_7/StatefulPartitionedCallStatefulPartitionedCall?tf.image.resize_3/resize/ResizeNearestNeighbor:resized_images:0conv2d_7_3228conv2d_7_3230*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:€€€€€€€€€АА*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_conv2d_7_layer_call_and_return_conditional_losses_2088д
reshape/PartitionedCallPartitionedCall)conv2d_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:€€€€€€€€€АА* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_reshape_layer_call_and_return_conditional_losses_2108y
IdentityIdentity reshape/PartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:€€€€€€€€€АА™
NoOpNoOp,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall.^batch_normalization_3/StatefulPartitionedCall.^batch_normalization_4/StatefulPartitionedCall.^batch_normalization_5/StatefulPartitionedCall.^batch_normalization_6/StatefulPartitionedCall^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall!^conv2d_4/StatefulPartitionedCall!^conv2d_5/StatefulPartitionedCall!^conv2d_6/StatefulPartitionedCall!^conv2d_7/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Ґ
_input_shapesР
Н:€€€€€€€€€АА: : : : : : : : : : : : : : : : : : : : : : : : :: : : : : : :: : : : : : :: : : : : : :: : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2^
-batch_normalization_3/StatefulPartitionedCall-batch_normalization_3/StatefulPartitionedCall2^
-batch_normalization_4/StatefulPartitionedCall-batch_normalization_4/StatefulPartitionedCall2^
-batch_normalization_5/StatefulPartitionedCall-batch_normalization_5/StatefulPartitionedCall2^
-batch_normalization_6/StatefulPartitionedCall-batch_normalization_6/StatefulPartitionedCall2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall2D
 conv2d_5/StatefulPartitionedCall conv2d_5/StatefulPartitionedCall2D
 conv2d_6/StatefulPartitionedCall conv2d_6/StatefulPartitionedCall2D
 conv2d_7/StatefulPartitionedCall conv2d_7/StatefulPartitionedCall:Z V
1
_output_shapes
:€€€€€€€€€АА
!
_user_specified_name	input_1: 

_output_shapes
::  

_output_shapes
:: '

_output_shapes
:: .

_output_shapes
:
Р
Њ
O__inference_batch_normalization_5_layer_call_and_return_conditional_losses_2325

inputs%
readvariableop_resource:`'
readvariableop_1_resource:`6
(fusedbatchnormv3_readvariableop_resource:`8
*fusedbatchnormv3_readvariableop_1_resource:`
identityИҐAssignNewValueҐAssignNewValue_1ҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:`*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:`*
dtype0Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:`*
dtype0И
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:`*
dtype0ƒ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€  `:`:`:`:`:*
epsilon%oГ:*
exponential_avg_factor%
„#<∞
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0Ї
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0k
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:€€€€€€€€€  `‘
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:€€€€€€€€€  `: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:€€€€€€€€€  `
 
_user_specified_nameinputs
В
Ъ
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_4343

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identityИҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0И
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0ґ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€@@ : : : : :*
epsilon%oГ:*
is_training( k
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:€€€€€€€€€@@ ∞
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:€€€€€€€€€@@ : : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:€€€€€€€€€@@ 
 
_user_specified_nameinputs
≠

щ
@__inference_conv2d_layer_call_and_return_conditional_losses_1699

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identityИҐBiasAdd/ReadVariableOpҐConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0Ы
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:€€€€€€€€€АА*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:€€€€€€€€€ААi
IdentityIdentityBiasAdd:output:0^NoOp*
T0*1
_output_shapes
:€€€€€€€€€ААw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:€€€€€€€€€АА: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:€€€€€€€€€АА
 
_user_specified_nameinputs
Т
Ю
O__inference_batch_normalization_3_layer_call_and_return_conditional_losses_4689

inputs&
readvariableop_resource:	А(
readvariableop_1_resource:	А7
(fusedbatchnormv3_readvariableop_resource:	А9
*fusedbatchnormv3_readvariableop_1_resource:	А
identityИҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:А*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:А*
dtype0Е
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype0Й
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype0ї
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:€€€€€€€€€А:А:А:А:А:*
epsilon%oГ:*
is_training( l
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*0
_output_shapes
:€€€€€€€€€А∞
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:€€€€€€€€€А: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:X T
0
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
Р
Њ
O__inference_batch_normalization_5_layer_call_and_return_conditional_losses_5033

inputs%
readvariableop_resource:`'
readvariableop_1_resource:`6
(fusedbatchnormv3_readvariableop_resource:`8
*fusedbatchnormv3_readvariableop_1_resource:`
identityИҐAssignNewValueҐAssignNewValue_1ҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:`*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:`*
dtype0Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:`*
dtype0И
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:`*
dtype0ƒ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€  `:`:`:`:`:*
epsilon%oГ:*
exponential_avg_factor%
„#<∞
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0Ї
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0k
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:€€€€€€€€€  `‘
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:€€€€€€€€€  `: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:€€€€€€€€€  `
 
_user_specified_nameinputs
«
ѕ
4__inference_batch_normalization_2_layer_call_fn_4449

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identityИҐStatefulPartitionedCallЙ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€  @*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *X
fSRQ
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1834w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:€€€€€€€€€  @`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:€€€€€€€€€  @: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:€€€€€€€€€  @
 
_user_specified_nameinputs
й
]
A__inference_re_lu_3_layer_call_and_return_conditional_losses_4717

inputs
identityO
ReluReluinputs*
T0*0
_output_shapes
:€€€€€€€€€Аc
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:€€€€€€€€€А"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:€€€€€€€€€А:X T
0
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
ђ

э
B__inference_conv2d_3_layer_call_and_return_conditional_losses_4583

inputs9
conv2d_readvariableop_resource:@А.
biasadd_readvariableop_resource:	А
identityИҐBiasAdd/ReadVariableOpҐConv2D/ReadVariableOp}
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@А*
dtype0Ъ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€Аh
IdentityIdentityBiasAdd:output:0^NoOp*
T0*0
_output_shapes
:€€€€€€€€€Аw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:€€€€€€€€€@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:€€€€€€€€€@
 
_user_specified_nameinputs
∞

ю
B__inference_conv2d_4_layer_call_and_return_conditional_losses_4756

inputs:
conv2d_readvariableop_resource:јј.
biasadd_readvariableop_resource:	ј
identityИҐBiasAdd/ReadVariableOpҐConv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:јј*
dtype0Ъ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€ј*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:ј*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€јh
IdentityIdentityBiasAdd:output:0^NoOp*
T0*0
_output_shapes
:€€€€€€€€€јw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :€€€€€€€€€ј: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:€€€€€€€€€ј
 
_user_specified_nameinputs
ЮО
Г0
__inference__wrapped_model_1186
input_1K
1mobile_unet_conv2d_conv2d_readvariableop_resource:@
2mobile_unet_conv2d_biasadd_readvariableop_resource:E
7mobile_unet_batch_normalization_readvariableop_resource:G
9mobile_unet_batch_normalization_readvariableop_1_resource:V
Hmobile_unet_batch_normalization_fusedbatchnormv3_readvariableop_resource:X
Jmobile_unet_batch_normalization_fusedbatchnormv3_readvariableop_1_resource:M
3mobile_unet_conv2d_1_conv2d_readvariableop_resource: B
4mobile_unet_conv2d_1_biasadd_readvariableop_resource: G
9mobile_unet_batch_normalization_1_readvariableop_resource: I
;mobile_unet_batch_normalization_1_readvariableop_1_resource: X
Jmobile_unet_batch_normalization_1_fusedbatchnormv3_readvariableop_resource: Z
Lmobile_unet_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource: M
3mobile_unet_conv2d_2_conv2d_readvariableop_resource: @B
4mobile_unet_conv2d_2_biasadd_readvariableop_resource:@G
9mobile_unet_batch_normalization_2_readvariableop_resource:@I
;mobile_unet_batch_normalization_2_readvariableop_1_resource:@X
Jmobile_unet_batch_normalization_2_fusedbatchnormv3_readvariableop_resource:@Z
Lmobile_unet_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource:@N
3mobile_unet_conv2d_3_conv2d_readvariableop_resource:@АC
4mobile_unet_conv2d_3_biasadd_readvariableop_resource:	АH
9mobile_unet_batch_normalization_3_readvariableop_resource:	АJ
;mobile_unet_batch_normalization_3_readvariableop_1_resource:	АY
Jmobile_unet_batch_normalization_3_fusedbatchnormv3_readvariableop_resource:	А[
Lmobile_unet_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource:	А+
'mobile_unet_tf_image_resize_resize_sizeO
3mobile_unet_conv2d_4_conv2d_readvariableop_resource:јјC
4mobile_unet_conv2d_4_biasadd_readvariableop_resource:	јH
9mobile_unet_batch_normalization_4_readvariableop_resource:	јJ
;mobile_unet_batch_normalization_4_readvariableop_1_resource:	јY
Jmobile_unet_batch_normalization_4_fusedbatchnormv3_readvariableop_resource:	ј[
Lmobile_unet_batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource:	ј-
)mobile_unet_tf_image_resize_1_resize_sizeN
3mobile_unet_conv2d_5_conv2d_readvariableop_resource:а`B
4mobile_unet_conv2d_5_biasadd_readvariableop_resource:`G
9mobile_unet_batch_normalization_5_readvariableop_resource:`I
;mobile_unet_batch_normalization_5_readvariableop_1_resource:`X
Jmobile_unet_batch_normalization_5_fusedbatchnormv3_readvariableop_resource:`Z
Lmobile_unet_batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource:`-
)mobile_unet_tf_image_resize_2_resize_sizeM
3mobile_unet_conv2d_6_conv2d_readvariableop_resource:p0B
4mobile_unet_conv2d_6_biasadd_readvariableop_resource:0G
9mobile_unet_batch_normalization_6_readvariableop_resource:0I
;mobile_unet_batch_normalization_6_readvariableop_1_resource:0X
Jmobile_unet_batch_normalization_6_fusedbatchnormv3_readvariableop_resource:0Z
Lmobile_unet_batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource:0-
)mobile_unet_tf_image_resize_3_resize_sizeM
3mobile_unet_conv2d_7_conv2d_readvariableop_resource:0B
4mobile_unet_conv2d_7_biasadd_readvariableop_resource:
identityИҐ?Mobile_Unet/batch_normalization/FusedBatchNormV3/ReadVariableOpҐAMobile_Unet/batch_normalization/FusedBatchNormV3/ReadVariableOp_1Ґ.Mobile_Unet/batch_normalization/ReadVariableOpҐ0Mobile_Unet/batch_normalization/ReadVariableOp_1ҐAMobile_Unet/batch_normalization_1/FusedBatchNormV3/ReadVariableOpҐCMobile_Unet/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1Ґ0Mobile_Unet/batch_normalization_1/ReadVariableOpҐ2Mobile_Unet/batch_normalization_1/ReadVariableOp_1ҐAMobile_Unet/batch_normalization_2/FusedBatchNormV3/ReadVariableOpҐCMobile_Unet/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1Ґ0Mobile_Unet/batch_normalization_2/ReadVariableOpҐ2Mobile_Unet/batch_normalization_2/ReadVariableOp_1ҐAMobile_Unet/batch_normalization_3/FusedBatchNormV3/ReadVariableOpҐCMobile_Unet/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1Ґ0Mobile_Unet/batch_normalization_3/ReadVariableOpҐ2Mobile_Unet/batch_normalization_3/ReadVariableOp_1ҐAMobile_Unet/batch_normalization_4/FusedBatchNormV3/ReadVariableOpҐCMobile_Unet/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1Ґ0Mobile_Unet/batch_normalization_4/ReadVariableOpҐ2Mobile_Unet/batch_normalization_4/ReadVariableOp_1ҐAMobile_Unet/batch_normalization_5/FusedBatchNormV3/ReadVariableOpҐCMobile_Unet/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1Ґ0Mobile_Unet/batch_normalization_5/ReadVariableOpҐ2Mobile_Unet/batch_normalization_5/ReadVariableOp_1ҐAMobile_Unet/batch_normalization_6/FusedBatchNormV3/ReadVariableOpҐCMobile_Unet/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1Ґ0Mobile_Unet/batch_normalization_6/ReadVariableOpҐ2Mobile_Unet/batch_normalization_6/ReadVariableOp_1Ґ)Mobile_Unet/conv2d/BiasAdd/ReadVariableOpҐ(Mobile_Unet/conv2d/Conv2D/ReadVariableOpҐ+Mobile_Unet/conv2d_1/BiasAdd/ReadVariableOpҐ*Mobile_Unet/conv2d_1/Conv2D/ReadVariableOpҐ+Mobile_Unet/conv2d_2/BiasAdd/ReadVariableOpҐ*Mobile_Unet/conv2d_2/Conv2D/ReadVariableOpҐ+Mobile_Unet/conv2d_3/BiasAdd/ReadVariableOpҐ*Mobile_Unet/conv2d_3/Conv2D/ReadVariableOpҐ+Mobile_Unet/conv2d_4/BiasAdd/ReadVariableOpҐ*Mobile_Unet/conv2d_4/Conv2D/ReadVariableOpҐ+Mobile_Unet/conv2d_5/BiasAdd/ReadVariableOpҐ*Mobile_Unet/conv2d_5/Conv2D/ReadVariableOpҐ+Mobile_Unet/conv2d_6/BiasAdd/ReadVariableOpҐ*Mobile_Unet/conv2d_6/Conv2D/ReadVariableOpҐ+Mobile_Unet/conv2d_7/BiasAdd/ReadVariableOpҐ*Mobile_Unet/conv2d_7/Conv2D/ReadVariableOpҐ
(Mobile_Unet/conv2d/Conv2D/ReadVariableOpReadVariableOp1mobile_unet_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0¬
Mobile_Unet/conv2d/Conv2DConv2Dinput_10Mobile_Unet/conv2d/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:€€€€€€€€€АА*
paddingSAME*
strides
Ш
)Mobile_Unet/conv2d/BiasAdd/ReadVariableOpReadVariableOp2mobile_unet_conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Є
Mobile_Unet/conv2d/BiasAddBiasAdd"Mobile_Unet/conv2d/Conv2D:output:01Mobile_Unet/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:€€€€€€€€€ААҐ
.Mobile_Unet/batch_normalization/ReadVariableOpReadVariableOp7mobile_unet_batch_normalization_readvariableop_resource*
_output_shapes
:*
dtype0¶
0Mobile_Unet/batch_normalization/ReadVariableOp_1ReadVariableOp9mobile_unet_batch_normalization_readvariableop_1_resource*
_output_shapes
:*
dtype0ƒ
?Mobile_Unet/batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOpHmobile_unet_batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0»
AMobile_Unet/batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpJmobile_unet_batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0х
0Mobile_Unet/batch_normalization/FusedBatchNormV3FusedBatchNormV3#Mobile_Unet/conv2d/BiasAdd:output:06Mobile_Unet/batch_normalization/ReadVariableOp:value:08Mobile_Unet/batch_normalization/ReadVariableOp_1:value:0GMobile_Unet/batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0IMobile_Unet/batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:€€€€€€€€€АА:::::*
epsilon%oГ:*
is_training( Р
Mobile_Unet/re_lu/ReluRelu4Mobile_Unet/batch_normalization/FusedBatchNormV3:y:0*
T0*1
_output_shapes
:€€€€€€€€€ААЊ
!Mobile_Unet/max_pooling2d/MaxPoolMaxPool$Mobile_Unet/re_lu/Relu:activations:0*/
_output_shapes
:€€€€€€€€€@@*
ksize
*
paddingSAME*
strides
¶
*Mobile_Unet/conv2d_1/Conv2D/ReadVariableOpReadVariableOp3mobile_unet_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0з
Mobile_Unet/conv2d_1/Conv2DConv2D*Mobile_Unet/max_pooling2d/MaxPool:output:02Mobile_Unet/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€@@ *
paddingSAME*
strides
Ь
+Mobile_Unet/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp4mobile_unet_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0Љ
Mobile_Unet/conv2d_1/BiasAddBiasAdd$Mobile_Unet/conv2d_1/Conv2D:output:03Mobile_Unet/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€@@ ¶
0Mobile_Unet/batch_normalization_1/ReadVariableOpReadVariableOp9mobile_unet_batch_normalization_1_readvariableop_resource*
_output_shapes
: *
dtype0™
2Mobile_Unet/batch_normalization_1/ReadVariableOp_1ReadVariableOp;mobile_unet_batch_normalization_1_readvariableop_1_resource*
_output_shapes
: *
dtype0»
AMobile_Unet/batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOpJmobile_unet_batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0ћ
CMobile_Unet/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpLmobile_unet_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0€
2Mobile_Unet/batch_normalization_1/FusedBatchNormV3FusedBatchNormV3%Mobile_Unet/conv2d_1/BiasAdd:output:08Mobile_Unet/batch_normalization_1/ReadVariableOp:value:0:Mobile_Unet/batch_normalization_1/ReadVariableOp_1:value:0IMobile_Unet/batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0KMobile_Unet/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€@@ : : : : :*
epsilon%oГ:*
is_training( Т
Mobile_Unet/re_lu_1/ReluRelu6Mobile_Unet/batch_normalization_1/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:€€€€€€€€€@@ ¬
#Mobile_Unet/max_pooling2d_1/MaxPoolMaxPool&Mobile_Unet/re_lu_1/Relu:activations:0*/
_output_shapes
:€€€€€€€€€   *
ksize
*
paddingSAME*
strides
¶
*Mobile_Unet/conv2d_2/Conv2D/ReadVariableOpReadVariableOp3mobile_unet_conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0й
Mobile_Unet/conv2d_2/Conv2DConv2D,Mobile_Unet/max_pooling2d_1/MaxPool:output:02Mobile_Unet/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€  @*
paddingSAME*
strides
Ь
+Mobile_Unet/conv2d_2/BiasAdd/ReadVariableOpReadVariableOp4mobile_unet_conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Љ
Mobile_Unet/conv2d_2/BiasAddBiasAdd$Mobile_Unet/conv2d_2/Conv2D:output:03Mobile_Unet/conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€  @¶
0Mobile_Unet/batch_normalization_2/ReadVariableOpReadVariableOp9mobile_unet_batch_normalization_2_readvariableop_resource*
_output_shapes
:@*
dtype0™
2Mobile_Unet/batch_normalization_2/ReadVariableOp_1ReadVariableOp;mobile_unet_batch_normalization_2_readvariableop_1_resource*
_output_shapes
:@*
dtype0»
AMobile_Unet/batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOpJmobile_unet_batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0ћ
CMobile_Unet/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpLmobile_unet_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0€
2Mobile_Unet/batch_normalization_2/FusedBatchNormV3FusedBatchNormV3%Mobile_Unet/conv2d_2/BiasAdd:output:08Mobile_Unet/batch_normalization_2/ReadVariableOp:value:0:Mobile_Unet/batch_normalization_2/ReadVariableOp_1:value:0IMobile_Unet/batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0KMobile_Unet/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€  @:@:@:@:@:*
epsilon%oГ:*
is_training( Т
Mobile_Unet/re_lu_2/ReluRelu6Mobile_Unet/batch_normalization_2/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:€€€€€€€€€  @¬
#Mobile_Unet/max_pooling2d_2/MaxPoolMaxPool&Mobile_Unet/re_lu_2/Relu:activations:0*/
_output_shapes
:€€€€€€€€€@*
ksize
*
paddingSAME*
strides
І
*Mobile_Unet/conv2d_3/Conv2D/ReadVariableOpReadVariableOp3mobile_unet_conv2d_3_conv2d_readvariableop_resource*'
_output_shapes
:@А*
dtype0к
Mobile_Unet/conv2d_3/Conv2DConv2D,Mobile_Unet/max_pooling2d_2/MaxPool:output:02Mobile_Unet/conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А*
paddingSAME*
strides
Э
+Mobile_Unet/conv2d_3/BiasAdd/ReadVariableOpReadVariableOp4mobile_unet_conv2d_3_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0љ
Mobile_Unet/conv2d_3/BiasAddBiasAdd$Mobile_Unet/conv2d_3/Conv2D:output:03Mobile_Unet/conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€АІ
0Mobile_Unet/batch_normalization_3/ReadVariableOpReadVariableOp9mobile_unet_batch_normalization_3_readvariableop_resource*
_output_shapes	
:А*
dtype0Ђ
2Mobile_Unet/batch_normalization_3/ReadVariableOp_1ReadVariableOp;mobile_unet_batch_normalization_3_readvariableop_1_resource*
_output_shapes	
:А*
dtype0…
AMobile_Unet/batch_normalization_3/FusedBatchNormV3/ReadVariableOpReadVariableOpJmobile_unet_batch_normalization_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype0Ќ
CMobile_Unet/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpLmobile_unet_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype0Д
2Mobile_Unet/batch_normalization_3/FusedBatchNormV3FusedBatchNormV3%Mobile_Unet/conv2d_3/BiasAdd:output:08Mobile_Unet/batch_normalization_3/ReadVariableOp:value:0:Mobile_Unet/batch_normalization_3/ReadVariableOp_1:value:0IMobile_Unet/batch_normalization_3/FusedBatchNormV3/ReadVariableOp:value:0KMobile_Unet/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:€€€€€€€€€А:А:А:А:А:*
epsilon%oГ:*
is_training( У
Mobile_Unet/re_lu_3/ReluRelu6Mobile_Unet/batch_normalization_3/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:€€€€€€€€€А√
#Mobile_Unet/max_pooling2d_3/MaxPoolMaxPool&Mobile_Unet/re_lu_3/Relu:activations:0*0
_output_shapes
:€€€€€€€€€А*
ksize
*
paddingSAME*
strides
э
8Mobile_Unet/tf.image.resize/resize/ResizeNearestNeighborResizeNearestNeighbor,Mobile_Unet/max_pooling2d_3/MaxPool:output:0'mobile_unet_tf_image_resize_resize_size*
T0*0
_output_shapes
:€€€€€€€€€А*
half_pixel_centers(c
!Mobile_Unet/tf.concat/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :С
Mobile_Unet/tf.concat/concatConcatV2IMobile_Unet/tf.image.resize/resize/ResizeNearestNeighbor:resized_images:0,Mobile_Unet/max_pooling2d_2/MaxPool:output:0*Mobile_Unet/tf.concat/concat/axis:output:0*
N*
T0*0
_output_shapes
:€€€€€€€€€ј®
*Mobile_Unet/conv2d_4/Conv2D/ReadVariableOpReadVariableOp3mobile_unet_conv2d_4_conv2d_readvariableop_resource*(
_output_shapes
:јј*
dtype0г
Mobile_Unet/conv2d_4/Conv2DConv2D%Mobile_Unet/tf.concat/concat:output:02Mobile_Unet/conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€ј*
paddingSAME*
strides
Э
+Mobile_Unet/conv2d_4/BiasAdd/ReadVariableOpReadVariableOp4mobile_unet_conv2d_4_biasadd_readvariableop_resource*
_output_shapes	
:ј*
dtype0љ
Mobile_Unet/conv2d_4/BiasAddBiasAdd$Mobile_Unet/conv2d_4/Conv2D:output:03Mobile_Unet/conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€јІ
0Mobile_Unet/batch_normalization_4/ReadVariableOpReadVariableOp9mobile_unet_batch_normalization_4_readvariableop_resource*
_output_shapes	
:ј*
dtype0Ђ
2Mobile_Unet/batch_normalization_4/ReadVariableOp_1ReadVariableOp;mobile_unet_batch_normalization_4_readvariableop_1_resource*
_output_shapes	
:ј*
dtype0…
AMobile_Unet/batch_normalization_4/FusedBatchNormV3/ReadVariableOpReadVariableOpJmobile_unet_batch_normalization_4_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:ј*
dtype0Ќ
CMobile_Unet/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpLmobile_unet_batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:ј*
dtype0Д
2Mobile_Unet/batch_normalization_4/FusedBatchNormV3FusedBatchNormV3%Mobile_Unet/conv2d_4/BiasAdd:output:08Mobile_Unet/batch_normalization_4/ReadVariableOp:value:0:Mobile_Unet/batch_normalization_4/ReadVariableOp_1:value:0IMobile_Unet/batch_normalization_4/FusedBatchNormV3/ReadVariableOp:value:0KMobile_Unet/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:€€€€€€€€€ј:ј:ј:ј:ј:*
epsilon%oГ:*
is_training( У
Mobile_Unet/re_lu_4/ReluRelu6Mobile_Unet/batch_normalization_4/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:€€€€€€€€€јы
:Mobile_Unet/tf.image.resize_1/resize/ResizeNearestNeighborResizeNearestNeighbor&Mobile_Unet/re_lu_4/Relu:activations:0)mobile_unet_tf_image_resize_1_resize_size*
T0*0
_output_shapes
:€€€€€€€€€  ј*
half_pixel_centers(e
#Mobile_Unet/tf.concat_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :Ч
Mobile_Unet/tf.concat_1/concatConcatV2KMobile_Unet/tf.image.resize_1/resize/ResizeNearestNeighbor:resized_images:0,Mobile_Unet/max_pooling2d_1/MaxPool:output:0,Mobile_Unet/tf.concat_1/concat/axis:output:0*
N*
T0*0
_output_shapes
:€€€€€€€€€  аІ
*Mobile_Unet/conv2d_5/Conv2D/ReadVariableOpReadVariableOp3mobile_unet_conv2d_5_conv2d_readvariableop_resource*'
_output_shapes
:а`*
dtype0д
Mobile_Unet/conv2d_5/Conv2DConv2D'Mobile_Unet/tf.concat_1/concat:output:02Mobile_Unet/conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€  `*
paddingSAME*
strides
Ь
+Mobile_Unet/conv2d_5/BiasAdd/ReadVariableOpReadVariableOp4mobile_unet_conv2d_5_biasadd_readvariableop_resource*
_output_shapes
:`*
dtype0Љ
Mobile_Unet/conv2d_5/BiasAddBiasAdd$Mobile_Unet/conv2d_5/Conv2D:output:03Mobile_Unet/conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€  `¶
0Mobile_Unet/batch_normalization_5/ReadVariableOpReadVariableOp9mobile_unet_batch_normalization_5_readvariableop_resource*
_output_shapes
:`*
dtype0™
2Mobile_Unet/batch_normalization_5/ReadVariableOp_1ReadVariableOp;mobile_unet_batch_normalization_5_readvariableop_1_resource*
_output_shapes
:`*
dtype0»
AMobile_Unet/batch_normalization_5/FusedBatchNormV3/ReadVariableOpReadVariableOpJmobile_unet_batch_normalization_5_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:`*
dtype0ћ
CMobile_Unet/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpLmobile_unet_batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:`*
dtype0€
2Mobile_Unet/batch_normalization_5/FusedBatchNormV3FusedBatchNormV3%Mobile_Unet/conv2d_5/BiasAdd:output:08Mobile_Unet/batch_normalization_5/ReadVariableOp:value:0:Mobile_Unet/batch_normalization_5/ReadVariableOp_1:value:0IMobile_Unet/batch_normalization_5/FusedBatchNormV3/ReadVariableOp:value:0KMobile_Unet/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€  `:`:`:`:`:*
epsilon%oГ:*
is_training( Т
Mobile_Unet/re_lu_5/ReluRelu6Mobile_Unet/batch_normalization_5/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:€€€€€€€€€  `ъ
:Mobile_Unet/tf.image.resize_2/resize/ResizeNearestNeighborResizeNearestNeighbor&Mobile_Unet/re_lu_5/Relu:activations:0)mobile_unet_tf_image_resize_2_resize_size*
T0*/
_output_shapes
:€€€€€€€€€@@`*
half_pixel_centers(e
#Mobile_Unet/tf.concat_2/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :Ф
Mobile_Unet/tf.concat_2/concatConcatV2KMobile_Unet/tf.image.resize_2/resize/ResizeNearestNeighbor:resized_images:0*Mobile_Unet/max_pooling2d/MaxPool:output:0,Mobile_Unet/tf.concat_2/concat/axis:output:0*
N*
T0*/
_output_shapes
:€€€€€€€€€@@p¶
*Mobile_Unet/conv2d_6/Conv2D/ReadVariableOpReadVariableOp3mobile_unet_conv2d_6_conv2d_readvariableop_resource*&
_output_shapes
:p0*
dtype0д
Mobile_Unet/conv2d_6/Conv2DConv2D'Mobile_Unet/tf.concat_2/concat:output:02Mobile_Unet/conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€@@0*
paddingSAME*
strides
Ь
+Mobile_Unet/conv2d_6/BiasAdd/ReadVariableOpReadVariableOp4mobile_unet_conv2d_6_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype0Љ
Mobile_Unet/conv2d_6/BiasAddBiasAdd$Mobile_Unet/conv2d_6/Conv2D:output:03Mobile_Unet/conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€@@0¶
0Mobile_Unet/batch_normalization_6/ReadVariableOpReadVariableOp9mobile_unet_batch_normalization_6_readvariableop_resource*
_output_shapes
:0*
dtype0™
2Mobile_Unet/batch_normalization_6/ReadVariableOp_1ReadVariableOp;mobile_unet_batch_normalization_6_readvariableop_1_resource*
_output_shapes
:0*
dtype0»
AMobile_Unet/batch_normalization_6/FusedBatchNormV3/ReadVariableOpReadVariableOpJmobile_unet_batch_normalization_6_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype0ћ
CMobile_Unet/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpLmobile_unet_batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype0€
2Mobile_Unet/batch_normalization_6/FusedBatchNormV3FusedBatchNormV3%Mobile_Unet/conv2d_6/BiasAdd:output:08Mobile_Unet/batch_normalization_6/ReadVariableOp:value:0:Mobile_Unet/batch_normalization_6/ReadVariableOp_1:value:0IMobile_Unet/batch_normalization_6/FusedBatchNormV3/ReadVariableOp:value:0KMobile_Unet/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€@@0:0:0:0:0:*
epsilon%oГ:*
is_training( Т
Mobile_Unet/re_lu_6/ReluRelu6Mobile_Unet/batch_normalization_6/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:€€€€€€€€€@@0ь
:Mobile_Unet/tf.image.resize_3/resize/ResizeNearestNeighborResizeNearestNeighbor&Mobile_Unet/re_lu_6/Relu:activations:0)mobile_unet_tf_image_resize_3_resize_size*
T0*1
_output_shapes
:€€€€€€€€€АА0*
half_pixel_centers(¶
*Mobile_Unet/conv2d_7/Conv2D/ReadVariableOpReadVariableOp3mobile_unet_conv2d_7_conv2d_readvariableop_resource*&
_output_shapes
:0*
dtype0К
Mobile_Unet/conv2d_7/Conv2DConv2DKMobile_Unet/tf.image.resize_3/resize/ResizeNearestNeighbor:resized_images:02Mobile_Unet/conv2d_7/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:€€€€€€€€€АА*
paddingSAME*
strides
Ь
+Mobile_Unet/conv2d_7/BiasAdd/ReadVariableOpReadVariableOp4mobile_unet_conv2d_7_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Њ
Mobile_Unet/conv2d_7/BiasAddBiasAdd$Mobile_Unet/conv2d_7/Conv2D:output:03Mobile_Unet/conv2d_7/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:€€€€€€€€€ААК
Mobile_Unet/conv2d_7/SigmoidSigmoid%Mobile_Unet/conv2d_7/BiasAdd:output:0*
T0*1
_output_shapes
:€€€€€€€€€ААi
Mobile_Unet/reshape/ShapeShape Mobile_Unet/conv2d_7/Sigmoid:y:0*
T0*
_output_shapes
:q
'Mobile_Unet/reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: s
)Mobile_Unet/reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:s
)Mobile_Unet/reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:µ
!Mobile_Unet/reshape/strided_sliceStridedSlice"Mobile_Unet/reshape/Shape:output:00Mobile_Unet/reshape/strided_slice/stack:output:02Mobile_Unet/reshape/strided_slice/stack_1:output:02Mobile_Unet/reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maske
#Mobile_Unet/reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :f
#Mobile_Unet/reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value
B :Аf
#Mobile_Unet/reshape/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value
B :АН
!Mobile_Unet/reshape/Reshape/shapePack*Mobile_Unet/reshape/strided_slice:output:0,Mobile_Unet/reshape/Reshape/shape/1:output:0,Mobile_Unet/reshape/Reshape/shape/2:output:0,Mobile_Unet/reshape/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:∞
Mobile_Unet/reshape/ReshapeReshape Mobile_Unet/conv2d_7/Sigmoid:y:0*Mobile_Unet/reshape/Reshape/shape:output:0*
T0*1
_output_shapes
:€€€€€€€€€АА}
IdentityIdentity$Mobile_Unet/reshape/Reshape:output:0^NoOp*
T0*1
_output_shapes
:€€€€€€€€€АА∞
NoOpNoOp@^Mobile_Unet/batch_normalization/FusedBatchNormV3/ReadVariableOpB^Mobile_Unet/batch_normalization/FusedBatchNormV3/ReadVariableOp_1/^Mobile_Unet/batch_normalization/ReadVariableOp1^Mobile_Unet/batch_normalization/ReadVariableOp_1B^Mobile_Unet/batch_normalization_1/FusedBatchNormV3/ReadVariableOpD^Mobile_Unet/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_11^Mobile_Unet/batch_normalization_1/ReadVariableOp3^Mobile_Unet/batch_normalization_1/ReadVariableOp_1B^Mobile_Unet/batch_normalization_2/FusedBatchNormV3/ReadVariableOpD^Mobile_Unet/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_11^Mobile_Unet/batch_normalization_2/ReadVariableOp3^Mobile_Unet/batch_normalization_2/ReadVariableOp_1B^Mobile_Unet/batch_normalization_3/FusedBatchNormV3/ReadVariableOpD^Mobile_Unet/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_11^Mobile_Unet/batch_normalization_3/ReadVariableOp3^Mobile_Unet/batch_normalization_3/ReadVariableOp_1B^Mobile_Unet/batch_normalization_4/FusedBatchNormV3/ReadVariableOpD^Mobile_Unet/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_11^Mobile_Unet/batch_normalization_4/ReadVariableOp3^Mobile_Unet/batch_normalization_4/ReadVariableOp_1B^Mobile_Unet/batch_normalization_5/FusedBatchNormV3/ReadVariableOpD^Mobile_Unet/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_11^Mobile_Unet/batch_normalization_5/ReadVariableOp3^Mobile_Unet/batch_normalization_5/ReadVariableOp_1B^Mobile_Unet/batch_normalization_6/FusedBatchNormV3/ReadVariableOpD^Mobile_Unet/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_11^Mobile_Unet/batch_normalization_6/ReadVariableOp3^Mobile_Unet/batch_normalization_6/ReadVariableOp_1*^Mobile_Unet/conv2d/BiasAdd/ReadVariableOp)^Mobile_Unet/conv2d/Conv2D/ReadVariableOp,^Mobile_Unet/conv2d_1/BiasAdd/ReadVariableOp+^Mobile_Unet/conv2d_1/Conv2D/ReadVariableOp,^Mobile_Unet/conv2d_2/BiasAdd/ReadVariableOp+^Mobile_Unet/conv2d_2/Conv2D/ReadVariableOp,^Mobile_Unet/conv2d_3/BiasAdd/ReadVariableOp+^Mobile_Unet/conv2d_3/Conv2D/ReadVariableOp,^Mobile_Unet/conv2d_4/BiasAdd/ReadVariableOp+^Mobile_Unet/conv2d_4/Conv2D/ReadVariableOp,^Mobile_Unet/conv2d_5/BiasAdd/ReadVariableOp+^Mobile_Unet/conv2d_5/Conv2D/ReadVariableOp,^Mobile_Unet/conv2d_6/BiasAdd/ReadVariableOp+^Mobile_Unet/conv2d_6/Conv2D/ReadVariableOp,^Mobile_Unet/conv2d_7/BiasAdd/ReadVariableOp+^Mobile_Unet/conv2d_7/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Ґ
_input_shapesР
Н:€€€€€€€€€АА: : : : : : : : : : : : : : : : : : : : : : : : :: : : : : : :: : : : : : :: : : : : : :: : 2В
?Mobile_Unet/batch_normalization/FusedBatchNormV3/ReadVariableOp?Mobile_Unet/batch_normalization/FusedBatchNormV3/ReadVariableOp2Ж
AMobile_Unet/batch_normalization/FusedBatchNormV3/ReadVariableOp_1AMobile_Unet/batch_normalization/FusedBatchNormV3/ReadVariableOp_12`
.Mobile_Unet/batch_normalization/ReadVariableOp.Mobile_Unet/batch_normalization/ReadVariableOp2d
0Mobile_Unet/batch_normalization/ReadVariableOp_10Mobile_Unet/batch_normalization/ReadVariableOp_12Ж
AMobile_Unet/batch_normalization_1/FusedBatchNormV3/ReadVariableOpAMobile_Unet/batch_normalization_1/FusedBatchNormV3/ReadVariableOp2К
CMobile_Unet/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1CMobile_Unet/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_12d
0Mobile_Unet/batch_normalization_1/ReadVariableOp0Mobile_Unet/batch_normalization_1/ReadVariableOp2h
2Mobile_Unet/batch_normalization_1/ReadVariableOp_12Mobile_Unet/batch_normalization_1/ReadVariableOp_12Ж
AMobile_Unet/batch_normalization_2/FusedBatchNormV3/ReadVariableOpAMobile_Unet/batch_normalization_2/FusedBatchNormV3/ReadVariableOp2К
CMobile_Unet/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1CMobile_Unet/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_12d
0Mobile_Unet/batch_normalization_2/ReadVariableOp0Mobile_Unet/batch_normalization_2/ReadVariableOp2h
2Mobile_Unet/batch_normalization_2/ReadVariableOp_12Mobile_Unet/batch_normalization_2/ReadVariableOp_12Ж
AMobile_Unet/batch_normalization_3/FusedBatchNormV3/ReadVariableOpAMobile_Unet/batch_normalization_3/FusedBatchNormV3/ReadVariableOp2К
CMobile_Unet/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1CMobile_Unet/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_12d
0Mobile_Unet/batch_normalization_3/ReadVariableOp0Mobile_Unet/batch_normalization_3/ReadVariableOp2h
2Mobile_Unet/batch_normalization_3/ReadVariableOp_12Mobile_Unet/batch_normalization_3/ReadVariableOp_12Ж
AMobile_Unet/batch_normalization_4/FusedBatchNormV3/ReadVariableOpAMobile_Unet/batch_normalization_4/FusedBatchNormV3/ReadVariableOp2К
CMobile_Unet/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1CMobile_Unet/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_12d
0Mobile_Unet/batch_normalization_4/ReadVariableOp0Mobile_Unet/batch_normalization_4/ReadVariableOp2h
2Mobile_Unet/batch_normalization_4/ReadVariableOp_12Mobile_Unet/batch_normalization_4/ReadVariableOp_12Ж
AMobile_Unet/batch_normalization_5/FusedBatchNormV3/ReadVariableOpAMobile_Unet/batch_normalization_5/FusedBatchNormV3/ReadVariableOp2К
CMobile_Unet/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1CMobile_Unet/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_12d
0Mobile_Unet/batch_normalization_5/ReadVariableOp0Mobile_Unet/batch_normalization_5/ReadVariableOp2h
2Mobile_Unet/batch_normalization_5/ReadVariableOp_12Mobile_Unet/batch_normalization_5/ReadVariableOp_12Ж
AMobile_Unet/batch_normalization_6/FusedBatchNormV3/ReadVariableOpAMobile_Unet/batch_normalization_6/FusedBatchNormV3/ReadVariableOp2К
CMobile_Unet/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1CMobile_Unet/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_12d
0Mobile_Unet/batch_normalization_6/ReadVariableOp0Mobile_Unet/batch_normalization_6/ReadVariableOp2h
2Mobile_Unet/batch_normalization_6/ReadVariableOp_12Mobile_Unet/batch_normalization_6/ReadVariableOp_12V
)Mobile_Unet/conv2d/BiasAdd/ReadVariableOp)Mobile_Unet/conv2d/BiasAdd/ReadVariableOp2T
(Mobile_Unet/conv2d/Conv2D/ReadVariableOp(Mobile_Unet/conv2d/Conv2D/ReadVariableOp2Z
+Mobile_Unet/conv2d_1/BiasAdd/ReadVariableOp+Mobile_Unet/conv2d_1/BiasAdd/ReadVariableOp2X
*Mobile_Unet/conv2d_1/Conv2D/ReadVariableOp*Mobile_Unet/conv2d_1/Conv2D/ReadVariableOp2Z
+Mobile_Unet/conv2d_2/BiasAdd/ReadVariableOp+Mobile_Unet/conv2d_2/BiasAdd/ReadVariableOp2X
*Mobile_Unet/conv2d_2/Conv2D/ReadVariableOp*Mobile_Unet/conv2d_2/Conv2D/ReadVariableOp2Z
+Mobile_Unet/conv2d_3/BiasAdd/ReadVariableOp+Mobile_Unet/conv2d_3/BiasAdd/ReadVariableOp2X
*Mobile_Unet/conv2d_3/Conv2D/ReadVariableOp*Mobile_Unet/conv2d_3/Conv2D/ReadVariableOp2Z
+Mobile_Unet/conv2d_4/BiasAdd/ReadVariableOp+Mobile_Unet/conv2d_4/BiasAdd/ReadVariableOp2X
*Mobile_Unet/conv2d_4/Conv2D/ReadVariableOp*Mobile_Unet/conv2d_4/Conv2D/ReadVariableOp2Z
+Mobile_Unet/conv2d_5/BiasAdd/ReadVariableOp+Mobile_Unet/conv2d_5/BiasAdd/ReadVariableOp2X
*Mobile_Unet/conv2d_5/Conv2D/ReadVariableOp*Mobile_Unet/conv2d_5/Conv2D/ReadVariableOp2Z
+Mobile_Unet/conv2d_6/BiasAdd/ReadVariableOp+Mobile_Unet/conv2d_6/BiasAdd/ReadVariableOp2X
*Mobile_Unet/conv2d_6/Conv2D/ReadVariableOp*Mobile_Unet/conv2d_6/Conv2D/ReadVariableOp2Z
+Mobile_Unet/conv2d_7/BiasAdd/ReadVariableOp+Mobile_Unet/conv2d_7/BiasAdd/ReadVariableOp2X
*Mobile_Unet/conv2d_7/Conv2D/ReadVariableOp*Mobile_Unet/conv2d_7/Conv2D/ReadVariableOp:Z V
1
_output_shapes
:€€€€€€€€€АА
!
_user_specified_name	input_1: 

_output_shapes
::  

_output_shapes
:: '

_output_shapes
:: .

_output_shapes
:
Р
e
I__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_1487

inputs
identity°
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€*
ksize
*
paddingSAME*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:r n
J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
ќ
J
.__inference_max_pooling2d_3_layer_call_fn_4727

inputs
identityј
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *R
fMRK
I__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_1911i
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:€€€€€€€€€А"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:€€€€€€€€€А:X T
0
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
Ц	
”
4__inference_batch_normalization_4_layer_call_fn_4782

inputs
unknown:	ј
	unknown_0:	ј
	unknown_1:	ј
	unknown_2:	ј
identityИҐStatefulPartitionedCallЪ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€ј*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *X
fSRQ
O__inference_batch_normalization_4_layer_call_and_return_conditional_losses_1543К
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€ј`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,€€€€€€€€€€€€€€€€€€€€€€€€€€€ј: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€ј
 
_user_specified_nameinputs
л
[
?__inference_re_lu_layer_call_and_return_conditional_losses_1737

inputs
identityP
ReluReluinputs*
T0*1
_output_shapes
:€€€€€€€€€ААd
IdentityIdentityRelu:activations:0*
T0*1
_output_shapes
:€€€€€€€€€АА"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:€€€€€€€€€АА:Y U
1
_output_shapes
:€€€€€€€€€АА
 
_user_specified_nameinputs
і
ь

*__inference_Mobile_Unet_layer_call_fn_3673

inputs!
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:#
	unknown_5: 
	unknown_6: 
	unknown_7: 
	unknown_8: 
	unknown_9: 

unknown_10: $

unknown_11: @

unknown_12:@

unknown_13:@

unknown_14:@

unknown_15:@

unknown_16:@%

unknown_17:@А

unknown_18:	А

unknown_19:	А

unknown_20:	А

unknown_21:	А

unknown_22:	А

unknown_23&

unknown_24:јј

unknown_25:	ј

unknown_26:	ј

unknown_27:	ј

unknown_28:	ј

unknown_29:	ј

unknown_30%

unknown_31:а`

unknown_32:`

unknown_33:`

unknown_34:`

unknown_35:`

unknown_36:`

unknown_37$

unknown_38:p0

unknown_39:0

unknown_40:0

unknown_41:0

unknown_42:0

unknown_43:0

unknown_44$

unknown_45:0

unknown_46:
identityИҐStatefulPartitionedCall–
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46*<
Tin5
321*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:€€€€€€€€€АА*@
_read_only_resource_inputs"
 	
!"#$()*+/0*0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_Mobile_Unet_layer_call_and_return_conditional_losses_2902y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:€€€€€€€€€АА`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Ґ
_input_shapesР
Н:€€€€€€€€€АА: : : : : : : : : : : : : : : : : : : : : : : : :: : : : : : :: : : : : : :: : : : : : :: : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:€€€€€€€€€АА
 
_user_specified_nameinputs: 

_output_shapes
::  

_output_shapes
:: '

_output_shapes
:: .

_output_shapes
:
й
]
A__inference_re_lu_3_layer_call_and_return_conditional_losses_1905

inputs
identityO
ReluReluinputs*
T0*0
_output_shapes
:€€€€€€€€€Аc
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:€€€€€€€€€А"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:€€€€€€€€€А:X T
0
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
Ќ
”
4__inference_batch_normalization_4_layer_call_fn_4808

inputs
unknown:	ј
	unknown_0:	ј
	unknown_1:	ј
	unknown_2:	ј
identityИҐStatefulPartitionedCallИ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€ј*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *X
fSRQ
O__inference_batch_normalization_4_layer_call_and_return_conditional_losses_2385x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:€€€€€€€€€ј`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:€€€€€€€€€ј: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:€€€€€€€€€ј
 
_user_specified_nameinputs
≈
ѕ
4__inference_batch_normalization_5_layer_call_fn_4961

inputs
unknown:`
	unknown_0:`
	unknown_1:`
	unknown_2:`
identityИҐStatefulPartitionedCallЗ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€  `*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *X
fSRQ
O__inference_batch_normalization_5_layer_call_and_return_conditional_losses_2325w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:€€€€€€€€€  ``
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:€€€€€€€€€  `: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:€€€€€€€€€  `
 
_user_specified_nameinputs
†
¬
O__inference_batch_normalization_4_layer_call_and_return_conditional_losses_4880

inputs&
readvariableop_resource:	ј(
readvariableop_1_resource:	ј7
(fusedbatchnormv3_readvariableop_resource:	ј9
*fusedbatchnormv3_readvariableop_1_resource:	ј
identityИҐAssignNewValueҐAssignNewValue_1ҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:ј*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:ј*
dtype0Е
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:ј*
dtype0Й
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:ј*
dtype0…
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:€€€€€€€€€ј:ј:ј:ј:ј:*
epsilon%oГ:*
exponential_avg_factor%
„#<∞
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0Ї
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0l
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*0
_output_shapes
:€€€€€€€€€ј‘
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:€€€€€€€€€ј: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:X T
0
_output_shapes
:€€€€€€€€€ј
 
_user_specified_nameinputs
Ї
B
&__inference_re_lu_2_layer_call_fn_4539

inputs
identityЈ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€  @* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_re_lu_2_layer_call_and_return_conditional_losses_1849h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:€€€€€€€€€  @"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:€€€€€€€€€  @:W S
/
_output_shapes
:€€€€€€€€€  @
 
_user_specified_nameinputs
Њ
B
&__inference_re_lu_4_layer_call_fn_4885

inputs
identityЄ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€ј* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_re_lu_4_layer_call_and_return_conditional_losses_1965i
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:€€€€€€€€€ј"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:€€€€€€€€€ј:X T
0
_output_shapes
:€€€€€€€€€ј
 
_user_specified_nameinputs
•

ы
B__inference_conv2d_6_layer_call_and_return_conditional_losses_2035

inputs8
conv2d_readvariableop_resource:p0-
biasadd_readvariableop_resource:0
identityИҐBiasAdd/ReadVariableOpҐConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:p0*
dtype0Щ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€@@0*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:0*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€@@0g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:€€€€€€€€€@@0w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:€€€€€€€€€@@p: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:€€€€€€€€€@@p
 
_user_specified_nameinputs
Ш	
”
4__inference_batch_normalization_4_layer_call_fn_4769

inputs
unknown:	ј
	unknown_0:	ј
	unknown_1:	ј
	unknown_2:	ј
identityИҐStatefulPartitionedCallЬ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€ј*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *X
fSRQ
O__inference_batch_normalization_4_layer_call_and_return_conditional_losses_1512К
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€ј`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,€€€€€€€€€€€€€€€€€€€€€€€€€€€ј: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€ј
 
_user_specified_nameinputs
≈
ѕ
4__inference_batch_normalization_1_layer_call_fn_4289

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identityИҐStatefulPartitionedCallЗ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€@@ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *X
fSRQ
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_2580w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:€€€€€€€€€@@ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:€€€€€€€€€@@ : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:€€€€€€€€€@@ 
 
_user_specified_nameinputs
е
]
A__inference_re_lu_5_layer_call_and_return_conditional_losses_5043

inputs
identityN
ReluReluinputs*
T0*/
_output_shapes
:€€€€€€€€€  `b
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:€€€€€€€€€  `"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:€€€€€€€€€  `:W S
/
_output_shapes
:€€€€€€€€€  `
 
_user_specified_nameinputs
й
Ь
'__inference_conv2d_1_layer_call_fn_4227

inputs!
unknown: 
	unknown_0: 
identityИҐStatefulPartitionedCallв
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€@@ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_conv2d_1_layer_call_and_return_conditional_losses_1755w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:€€€€€€€€€@@ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:€€€€€€€€€@@: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:€€€€€€€€€@@
 
_user_specified_nameinputs
Ј
э

*__inference_Mobile_Unet_layer_call_fn_3102
input_1!
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:#
	unknown_5: 
	unknown_6: 
	unknown_7: 
	unknown_8: 
	unknown_9: 

unknown_10: $

unknown_11: @

unknown_12:@

unknown_13:@

unknown_14:@

unknown_15:@

unknown_16:@%

unknown_17:@А

unknown_18:	А

unknown_19:	А

unknown_20:	А

unknown_21:	А

unknown_22:	А

unknown_23&

unknown_24:јј

unknown_25:	ј

unknown_26:	ј

unknown_27:	ј

unknown_28:	ј

unknown_29:	ј

unknown_30%

unknown_31:а`

unknown_32:`

unknown_33:`

unknown_34:`

unknown_35:`

unknown_36:`

unknown_37$

unknown_38:p0

unknown_39:0

unknown_40:0

unknown_41:0

unknown_42:0

unknown_43:0

unknown_44$

unknown_45:0

unknown_46:
identityИҐStatefulPartitionedCall—
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46*<
Tin5
321*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:€€€€€€€€€АА*@
_read_only_resource_inputs"
 	
!"#$()*+/0*0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_Mobile_Unet_layer_call_and_return_conditional_losses_2902y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:€€€€€€€€€АА`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Ґ
_input_shapesР
Н:€€€€€€€€€АА: : : : : : : : : : : : : : : : : : : : : : : : :: : : : : : :: : : : : : :: : : : : : :: : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
1
_output_shapes
:€€€€€€€€€АА
!
_user_specified_name	input_1: 

_output_shapes
::  

_output_shapes
:: '

_output_shapes
:: .

_output_shapes
:
Ї
B
&__inference_re_lu_5_layer_call_fn_5038

inputs
identityЈ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€  `* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_re_lu_5_layer_call_and_return_conditional_losses_2019h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:€€€€€€€€€  `"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:€€€€€€€€€  `:W S
/
_output_shapes
:€€€€€€€€€  `
 
_user_specified_nameinputs
е
]
A__inference_re_lu_2_layer_call_and_return_conditional_losses_1849

inputs
identityN
ReluReluinputs*
T0*/
_output_shapes
:€€€€€€€€€  @b
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:€€€€€€€€€  @"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:€€€€€€€€€  @:W S
/
_output_shapes
:€€€€€€€€€  @
 
_user_specified_nameinputs
Ў
Њ
O__inference_batch_normalization_5_layer_call_and_return_conditional_losses_1607

inputs%
readvariableop_resource:`'
readvariableop_1_resource:`6
(fusedbatchnormv3_readvariableop_resource:`8
*fusedbatchnormv3_readvariableop_1_resource:`
identityИҐAssignNewValueҐAssignNewValue_1ҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:`*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:`*
dtype0Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:`*
dtype0И
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:`*
dtype0÷
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€`:`:`:`:`:*
epsilon%oГ:*
exponential_avg_factor%
„#<∞
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0Ї
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€`‘
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+€€€€€€€€€€€€€€€€€€€€€€€€€€€`: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€`
 
_user_specified_nameinputs
Р	
ѕ
4__inference_batch_normalization_6_layer_call_fn_5075

inputs
unknown:0
	unknown_0:0
	unknown_1:0
	unknown_2:0
identityИҐStatefulPartitionedCallЫ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€0*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *X
fSRQ
O__inference_batch_normalization_6_layer_call_and_return_conditional_losses_1640Й
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€0`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+€€€€€€€€€€€€€€€€€€€€€€€€€€€0: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€0
 
_user_specified_nameinputs
®
e
I__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_1911

inputs
identityЗ
MaxPoolMaxPoolinputs*0
_output_shapes
:€€€€€€€€€А*
ksize
*
paddingSAME*
strides
a
IdentityIdentityMaxPool:output:0*
T0*0
_output_shapes
:€€€€€€€€€А"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:€€€€€€€€€А:X T
0
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
•

ы
B__inference_conv2d_2_layer_call_and_return_conditional_losses_4410

inputs8
conv2d_readvariableop_resource: @-
biasadd_readvariableop_resource:@
identityИҐBiasAdd/ReadVariableOpҐConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0Щ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€  @*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€  @g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:€€€€€€€€€  @w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:€€€€€€€€€   : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:€€€€€€€€€   
 
_user_specified_nameinputs
Ї
B
&__inference_re_lu_1_layer_call_fn_4366

inputs
identityЈ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€@@ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_re_lu_1_layer_call_and_return_conditional_losses_1793h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:€€€€€€€€€@@ "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:€€€€€€€€€@@ :W S
/
_output_shapes
:€€€€€€€€€@@ 
 
_user_specified_nameinputs
Р	
ѕ
4__inference_batch_normalization_5_layer_call_fn_4922

inputs
unknown:`
	unknown_0:`
	unknown_1:`
	unknown_2:`
identityИҐStatefulPartitionedCallЫ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€`*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *X
fSRQ
O__inference_batch_normalization_5_layer_call_and_return_conditional_losses_1576Й
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€``
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+€€€€€€€€€€€€€€€€€€€€€€€€€€€`: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€`
 
_user_specified_nameinputs
ђ

э
B__inference_conv2d_3_layer_call_and_return_conditional_losses_1867

inputs9
conv2d_readvariableop_resource:@А.
biasadd_readvariableop_resource:	А
identityИҐBiasAdd/ReadVariableOpҐConv2D/ReadVariableOp}
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@А*
dtype0Ъ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€Аh
IdentityIdentityBiasAdd:output:0^NoOp*
T0*0
_output_shapes
:€€€€€€€€€Аw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:€€€€€€€€€@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:€€€€€€€€€@
 
_user_specified_nameinputs
В
Ъ
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1778

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identityИҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0И
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0ґ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€@@ : : : : :*
epsilon%oГ:*
is_training( k
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:€€€€€€€€€@@ ∞
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:€€€€€€€€€@@ : : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:€€€€€€€€€@@ 
 
_user_specified_nameinputs
їО
Ь
E__inference_Mobile_Unet_layer_call_and_return_conditional_losses_2902

inputs%
conv2d_2772:
conv2d_2774:&
batch_normalization_2777:&
batch_normalization_2779:&
batch_normalization_2781:&
batch_normalization_2783:'
conv2d_1_2788: 
conv2d_1_2790: (
batch_normalization_1_2793: (
batch_normalization_1_2795: (
batch_normalization_1_2797: (
batch_normalization_1_2799: '
conv2d_2_2804: @
conv2d_2_2806:@(
batch_normalization_2_2809:@(
batch_normalization_2_2811:@(
batch_normalization_2_2813:@(
batch_normalization_2_2815:@(
conv2d_3_2820:@А
conv2d_3_2822:	А)
batch_normalization_3_2825:	А)
batch_normalization_3_2827:	А)
batch_normalization_3_2829:	А)
batch_normalization_3_2831:	А
tf_image_resize_resize_size)
conv2d_4_2840:јј
conv2d_4_2842:	ј)
batch_normalization_4_2845:	ј)
batch_normalization_4_2847:	ј)
batch_normalization_4_2849:	ј)
batch_normalization_4_2851:	ј!
tf_image_resize_1_resize_size(
conv2d_5_2859:а`
conv2d_5_2861:`(
batch_normalization_5_2864:`(
batch_normalization_5_2866:`(
batch_normalization_5_2868:`(
batch_normalization_5_2870:`!
tf_image_resize_2_resize_size'
conv2d_6_2878:p0
conv2d_6_2880:0(
batch_normalization_6_2883:0(
batch_normalization_6_2885:0(
batch_normalization_6_2887:0(
batch_normalization_6_2889:0!
tf_image_resize_3_resize_size'
conv2d_7_2895:0
conv2d_7_2897:
identityИҐ+batch_normalization/StatefulPartitionedCallҐ-batch_normalization_1/StatefulPartitionedCallҐ-batch_normalization_2/StatefulPartitionedCallҐ-batch_normalization_3/StatefulPartitionedCallҐ-batch_normalization_4/StatefulPartitionedCallҐ-batch_normalization_5/StatefulPartitionedCallҐ-batch_normalization_6/StatefulPartitionedCallҐconv2d/StatefulPartitionedCallҐ conv2d_1/StatefulPartitionedCallҐ conv2d_2/StatefulPartitionedCallҐ conv2d_3/StatefulPartitionedCallҐ conv2d_4/StatefulPartitionedCallҐ conv2d_5/StatefulPartitionedCallҐ conv2d_6/StatefulPartitionedCallҐ conv2d_7/StatefulPartitionedCallп
conv2d/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_2772conv2d_2774*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:€€€€€€€€€АА*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *I
fDRB
@__inference_conv2d_layer_call_and_return_conditional_losses_1699ъ
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0batch_normalization_2777batch_normalization_2779batch_normalization_2781batch_normalization_2783*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:€€€€€€€€€АА*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *V
fQRO
M__inference_batch_normalization_layer_call_and_return_conditional_losses_2645л
re_lu/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:€€€€€€€€€АА* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *H
fCRA
?__inference_re_lu_layer_call_and_return_conditional_losses_1737г
max_pooling2d/PartitionedCallPartitionedCallre_lu/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€@@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_max_pooling2d_layer_call_and_return_conditional_losses_1743Х
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0conv2d_1_2788conv2d_1_2790*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€@@ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_conv2d_1_layer_call_and_return_conditional_losses_1755Ж
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0batch_normalization_1_2793batch_normalization_1_2795batch_normalization_1_2797batch_normalization_1_2799*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€@@ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *X
fSRQ
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_2580п
re_lu_1/PartitionedCallPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€@@ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_re_lu_1_layer_call_and_return_conditional_losses_1793й
max_pooling2d_1/PartitionedCallPartitionedCall re_lu_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€   * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *R
fMRK
I__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_1799Ч
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_1/PartitionedCall:output:0conv2d_2_2804conv2d_2_2806*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€  @*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_conv2d_2_layer_call_and_return_conditional_losses_1811Ж
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0batch_normalization_2_2809batch_normalization_2_2811batch_normalization_2_2813batch_normalization_2_2815*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€  @*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *X
fSRQ
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_2515п
re_lu_2/PartitionedCallPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€  @* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_re_lu_2_layer_call_and_return_conditional_losses_1849й
max_pooling2d_2/PartitionedCallPartitionedCall re_lu_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *R
fMRK
I__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_1855Ш
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_2/PartitionedCall:output:0conv2d_3_2820conv2d_3_2822*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_conv2d_3_layer_call_and_return_conditional_losses_1867З
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0batch_normalization_3_2825batch_normalization_3_2827batch_normalization_3_2829batch_normalization_3_2831*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *X
fSRQ
O__inference_batch_normalization_3_layer_call_and_return_conditional_losses_2450р
re_lu_3/PartitionedCallPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_re_lu_3_layer_call_and_return_conditional_losses_1905к
max_pooling2d_3/PartitionedCallPartitionedCall re_lu_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *R
fMRK
I__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_1911б
,tf.image.resize/resize/ResizeNearestNeighborResizeNearestNeighbor(max_pooling2d_3/PartitionedCall:output:0tf_image_resize_resize_size*
T0*0
_output_shapes
:€€€€€€€€€А*
half_pixel_centers(W
tf.concat/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :й
tf.concat/concatConcatV2=tf.image.resize/resize/ResizeNearestNeighbor:resized_images:0(max_pooling2d_2/PartitionedCall:output:0tf.concat/concat/axis:output:0*
N*
T0*0
_output_shapes
:€€€€€€€€€јЙ
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCalltf.concat/concat:output:0conv2d_4_2840conv2d_4_2842*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€ј*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_conv2d_4_layer_call_and_return_conditional_losses_1927З
-batch_normalization_4/StatefulPartitionedCallStatefulPartitionedCall)conv2d_4/StatefulPartitionedCall:output:0batch_normalization_4_2845batch_normalization_4_2847batch_normalization_4_2849batch_normalization_4_2851*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€ј*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *X
fSRQ
O__inference_batch_normalization_4_layer_call_and_return_conditional_losses_2385р
re_lu_4/PartitionedCallPartitionedCall6batch_normalization_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€ј* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_re_lu_4_layer_call_and_return_conditional_losses_1965Ё
.tf.image.resize_1/resize/ResizeNearestNeighborResizeNearestNeighbor re_lu_4/PartitionedCall:output:0tf_image_resize_1_resize_size*
T0*0
_output_shapes
:€€€€€€€€€  ј*
half_pixel_centers(Y
tf.concat_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :п
tf.concat_1/concatConcatV2?tf.image.resize_1/resize/ResizeNearestNeighbor:resized_images:0(max_pooling2d_1/PartitionedCall:output:0 tf.concat_1/concat/axis:output:0*
N*
T0*0
_output_shapes
:€€€€€€€€€  аК
 conv2d_5/StatefulPartitionedCallStatefulPartitionedCalltf.concat_1/concat:output:0conv2d_5_2859conv2d_5_2861*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€  `*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_conv2d_5_layer_call_and_return_conditional_losses_1981Ж
-batch_normalization_5/StatefulPartitionedCallStatefulPartitionedCall)conv2d_5/StatefulPartitionedCall:output:0batch_normalization_5_2864batch_normalization_5_2866batch_normalization_5_2868batch_normalization_5_2870*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€  `*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *X
fSRQ
O__inference_batch_normalization_5_layer_call_and_return_conditional_losses_2325п
re_lu_5/PartitionedCallPartitionedCall6batch_normalization_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€  `* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_re_lu_5_layer_call_and_return_conditional_losses_2019№
.tf.image.resize_2/resize/ResizeNearestNeighborResizeNearestNeighbor re_lu_5/PartitionedCall:output:0tf_image_resize_2_resize_size*
T0*/
_output_shapes
:€€€€€€€€€@@`*
half_pixel_centers(Y
tf.concat_2/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :м
tf.concat_2/concatConcatV2?tf.image.resize_2/resize/ResizeNearestNeighbor:resized_images:0&max_pooling2d/PartitionedCall:output:0 tf.concat_2/concat/axis:output:0*
N*
T0*/
_output_shapes
:€€€€€€€€€@@pК
 conv2d_6/StatefulPartitionedCallStatefulPartitionedCalltf.concat_2/concat:output:0conv2d_6_2878conv2d_6_2880*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€@@0*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_conv2d_6_layer_call_and_return_conditional_losses_2035Ж
-batch_normalization_6/StatefulPartitionedCallStatefulPartitionedCall)conv2d_6/StatefulPartitionedCall:output:0batch_normalization_6_2883batch_normalization_6_2885batch_normalization_6_2887batch_normalization_6_2889*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€@@0*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *X
fSRQ
O__inference_batch_normalization_6_layer_call_and_return_conditional_losses_2265п
re_lu_6/PartitionedCallPartitionedCall6batch_normalization_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€@@0* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_re_lu_6_layer_call_and_return_conditional_losses_2073ё
.tf.image.resize_3/resize/ResizeNearestNeighborResizeNearestNeighbor re_lu_6/PartitionedCall:output:0tf_image_resize_3_resize_size*
T0*1
_output_shapes
:€€€€€€€€€АА0*
half_pixel_centers(∞
 conv2d_7/StatefulPartitionedCallStatefulPartitionedCall?tf.image.resize_3/resize/ResizeNearestNeighbor:resized_images:0conv2d_7_2895conv2d_7_2897*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:€€€€€€€€€АА*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_conv2d_7_layer_call_and_return_conditional_losses_2088д
reshape/PartitionedCallPartitionedCall)conv2d_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:€€€€€€€€€АА* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_reshape_layer_call_and_return_conditional_losses_2108y
IdentityIdentity reshape/PartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:€€€€€€€€€АА™
NoOpNoOp,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall.^batch_normalization_3/StatefulPartitionedCall.^batch_normalization_4/StatefulPartitionedCall.^batch_normalization_5/StatefulPartitionedCall.^batch_normalization_6/StatefulPartitionedCall^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall!^conv2d_4/StatefulPartitionedCall!^conv2d_5/StatefulPartitionedCall!^conv2d_6/StatefulPartitionedCall!^conv2d_7/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Ґ
_input_shapesР
Н:€€€€€€€€€АА: : : : : : : : : : : : : : : : : : : : : : : : :: : : : : : :: : : : : : :: : : : : : :: : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2^
-batch_normalization_3/StatefulPartitionedCall-batch_normalization_3/StatefulPartitionedCall2^
-batch_normalization_4/StatefulPartitionedCall-batch_normalization_4/StatefulPartitionedCall2^
-batch_normalization_5/StatefulPartitionedCall-batch_normalization_5/StatefulPartitionedCall2^
-batch_normalization_6/StatefulPartitionedCall-batch_normalization_6/StatefulPartitionedCall2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall2D
 conv2d_5/StatefulPartitionedCall conv2d_5/StatefulPartitionedCall2D
 conv2d_6/StatefulPartitionedCall conv2d_6/StatefulPartitionedCall2D
 conv2d_7/StatefulPartitionedCall conv2d_7/StatefulPartitionedCall:Y U
1
_output_shapes
:€€€€€€€€€АА
 
_user_specified_nameinputs: 

_output_shapes
::  

_output_shapes
:: '

_output_shapes
:: .

_output_shapes
:
М
ы
B__inference_conv2d_7_layer_call_and_return_conditional_losses_2088

inputs8
conv2d_readvariableop_resource:0-
biasadd_readvariableop_resource:
identityИҐBiasAdd/ReadVariableOpҐConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:0*
dtype0Ы
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:€€€€€€€€€АА*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:€€€€€€€€€АА`
SigmoidSigmoidBiasAdd:output:0*
T0*1
_output_shapes
:€€€€€€€€€ААd
IdentityIdentitySigmoid:y:0^NoOp*
T0*1
_output_shapes
:€€€€€€€€€ААw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:€€€€€€€€€АА0: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:€€€€€€€€€АА0
 
_user_specified_nameinputs
»
Ш
M__inference_batch_normalization_layer_call_and_return_conditional_losses_1208

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identityИҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0И
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0»
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€:::::*
epsilon%oГ:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€∞
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+€€€€€€€€€€€€€€€€€€€€€€€€€€€: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
≥
H
,__inference_max_pooling2d_layer_call_fn_4203

inputs
identityЎ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_max_pooling2d_layer_call_and_return_conditional_losses_1259Г
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:r n
J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
÷
Љ
M__inference_batch_normalization_layer_call_and_return_conditional_losses_1239

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identityИҐAssignNewValueҐAssignNewValue_1ҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0И
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0÷
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€:::::*
epsilon%oГ:*
exponential_avg_factor%
„#<∞
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0Ї
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€‘
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+€€€€€€€€€€€€€€€€€€€€€€€€€€€: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
Р
e
I__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_1411

inputs
identity°
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€*
ksize
*
paddingSAME*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:r n
J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
Ў
Њ
O__inference_batch_normalization_6_layer_call_and_return_conditional_losses_1671

inputs%
readvariableop_resource:0'
readvariableop_1_resource:06
(fusedbatchnormv3_readvariableop_resource:08
*fusedbatchnormv3_readvariableop_1_resource:0
identityИҐAssignNewValueҐAssignNewValue_1ҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:0*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:0*
dtype0Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype0И
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype0÷
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€0:0:0:0:0:*
epsilon%oГ:*
exponential_avg_factor%
„#<∞
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0Ї
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€0‘
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+€€€€€€€€€€€€€€€€€€€€€€€€€€€0: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€0
 
_user_specified_nameinputs
•

ы
B__inference_conv2d_6_layer_call_and_return_conditional_losses_5062

inputs8
conv2d_readvariableop_resource:p0-
biasadd_readvariableop_resource:0
identityИҐBiasAdd/ReadVariableOpҐConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:p0*
dtype0Щ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€@@0*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:0*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€@@0g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:€€€€€€€€€@@0w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:€€€€€€€€€@@p: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:€€€€€€€€€@@p
 
_user_specified_nameinputs
Р
e
I__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_4732

inputs
identity°
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€*
ksize
*
paddingSAME*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:r n
J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
Т
Ю
O__inference_batch_normalization_4_layer_call_and_return_conditional_losses_4862

inputs&
readvariableop_resource:	ј(
readvariableop_1_resource:	ј7
(fusedbatchnormv3_readvariableop_resource:	ј9
*fusedbatchnormv3_readvariableop_1_resource:	ј
identityИҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:ј*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:ј*
dtype0Е
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:ј*
dtype0Й
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:ј*
dtype0ї
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:€€€€€€€€€ј:ј:ј:ј:ј:*
epsilon%oГ:*
is_training( l
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*0
_output_shapes
:€€€€€€€€€ј∞
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:€€€€€€€€€ј: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:X T
0
_output_shapes
:€€€€€€€€€ј
 
_user_specified_nameinputs
О	
ѕ
4__inference_batch_normalization_1_layer_call_fn_4263

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identityИҐStatefulPartitionedCallЩ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *X
fSRQ
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1315Й
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 
 
_user_specified_nameinputs
¬
ь

*__inference_Mobile_Unet_layer_call_fn_3572

inputs!
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:#
	unknown_5: 
	unknown_6: 
	unknown_7: 
	unknown_8: 
	unknown_9: 

unknown_10: $

unknown_11: @

unknown_12:@

unknown_13:@

unknown_14:@

unknown_15:@

unknown_16:@%

unknown_17:@А

unknown_18:	А

unknown_19:	А

unknown_20:	А

unknown_21:	А

unknown_22:	А

unknown_23&

unknown_24:јј

unknown_25:	ј

unknown_26:	ј

unknown_27:	ј

unknown_28:	ј

unknown_29:	ј

unknown_30%

unknown_31:а`

unknown_32:`

unknown_33:`

unknown_34:`

unknown_35:`

unknown_36:`

unknown_37$

unknown_38:p0

unknown_39:0

unknown_40:0

unknown_41:0

unknown_42:0

unknown_43:0

unknown_44$

unknown_45:0

unknown_46:
identityИҐStatefulPartitionedCallё
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46*<
Tin5
321*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:€€€€€€€€€АА*N
_read_only_resource_inputs0
.,	
!"#$%&()*+,-/0*0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_Mobile_Unet_layer_call_and_return_conditional_losses_2111y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:€€€€€€€€€АА`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Ґ
_input_shapesР
Н:€€€€€€€€€АА: : : : : : : : : : : : : : : : : : : : : : : : :: : : : : : :: : : : : : :: : : : : : :: : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:€€€€€€€€€АА
 
_user_specified_nameinputs: 

_output_shapes
::  

_output_shapes
:: '

_output_shapes
:: .

_output_shapes
:
И
Ш
M__inference_batch_normalization_layer_call_and_return_conditional_losses_4170

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identityИҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0И
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0Є
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:€€€€€€€€€АА:::::*
epsilon%oГ:*
is_training( m
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*1
_output_shapes
:€€€€€€€€€АА∞
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:€€€€€€€€€АА: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Y U
1
_output_shapes
:€€€€€€€€€АА
 
_user_specified_nameinputs
»
Ш
M__inference_batch_normalization_layer_call_and_return_conditional_losses_4134

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identityИҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0И
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0»
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€:::::*
epsilon%oГ:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€∞
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+€€€€€€€€€€€€€€€€€€€€€€€€€€€: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
М
ы
B__inference_conv2d_7_layer_call_and_return_conditional_losses_5216

inputs8
conv2d_readvariableop_resource:0-
biasadd_readvariableop_resource:
identityИҐBiasAdd/ReadVariableOpҐConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:0*
dtype0Ы
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:€€€€€€€€€АА*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:€€€€€€€€€АА`
SigmoidSigmoidBiasAdd:output:0*
T0*1
_output_shapes
:€€€€€€€€€ААd
IdentityIdentitySigmoid:y:0^NoOp*
T0*1
_output_shapes
:€€€€€€€€€ААw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:€€€€€€€€€АА0: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:€€€€€€€€€АА0
 
_user_specified_nameinputs
В
Ъ
O__inference_batch_normalization_5_layer_call_and_return_conditional_losses_2004

inputs%
readvariableop_resource:`'
readvariableop_1_resource:`6
(fusedbatchnormv3_readvariableop_resource:`8
*fusedbatchnormv3_readvariableop_1_resource:`
identityИҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:`*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:`*
dtype0Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:`*
dtype0И
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:`*
dtype0ґ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€  `:`:`:`:`:*
epsilon%oГ:*
is_training( k
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:€€€€€€€€€  `∞
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:€€€€€€€€€  `: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:€€€€€€€€€  `
 
_user_specified_nameinputs
≈
ѕ
4__inference_batch_normalization_6_layer_call_fn_5114

inputs
unknown:0
	unknown_0:0
	unknown_1:0
	unknown_2:0
identityИҐStatefulPartitionedCallЗ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€@@0*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *X
fSRQ
O__inference_batch_normalization_6_layer_call_and_return_conditional_losses_2265w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:€€€€€€€€€@@0`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:€€€€€€€€€@@0: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:€€€€€€€€€@@0
 
_user_specified_nameinputs
В
Ъ
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_4516

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identityИҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0И
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0ґ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€  @:@:@:@:@:*
epsilon%oГ:*
is_training( k
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:€€€€€€€€€  @∞
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:€€€€€€€€€  @: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:€€€€€€€€€  @
 
_user_specified_nameinputs
 
Ъ
O__inference_batch_normalization_5_layer_call_and_return_conditional_losses_4979

inputs%
readvariableop_resource:`'
readvariableop_1_resource:`6
(fusedbatchnormv3_readvariableop_resource:`8
*fusedbatchnormv3_readvariableop_1_resource:`
identityИҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:`*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:`*
dtype0Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:`*
dtype0И
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:`*
dtype0»
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€`:`:`:`:`:*
epsilon%oГ:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€`∞
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+€€€€€€€€€€€€€€€€€€€€€€€€€€€`: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€`
 
_user_specified_nameinputs
Р
e
I__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_4559

inputs
identity°
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€*
ksize
*
paddingSAME*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:r n
J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
ЊО
Э
E__inference_Mobile_Unet_layer_call_and_return_conditional_losses_3368
input_1%
conv2d_3238:
conv2d_3240:&
batch_normalization_3243:&
batch_normalization_3245:&
batch_normalization_3247:&
batch_normalization_3249:'
conv2d_1_3254: 
conv2d_1_3256: (
batch_normalization_1_3259: (
batch_normalization_1_3261: (
batch_normalization_1_3263: (
batch_normalization_1_3265: '
conv2d_2_3270: @
conv2d_2_3272:@(
batch_normalization_2_3275:@(
batch_normalization_2_3277:@(
batch_normalization_2_3279:@(
batch_normalization_2_3281:@(
conv2d_3_3286:@А
conv2d_3_3288:	А)
batch_normalization_3_3291:	А)
batch_normalization_3_3293:	А)
batch_normalization_3_3295:	А)
batch_normalization_3_3297:	А
tf_image_resize_resize_size)
conv2d_4_3306:јј
conv2d_4_3308:	ј)
batch_normalization_4_3311:	ј)
batch_normalization_4_3313:	ј)
batch_normalization_4_3315:	ј)
batch_normalization_4_3317:	ј!
tf_image_resize_1_resize_size(
conv2d_5_3325:а`
conv2d_5_3327:`(
batch_normalization_5_3330:`(
batch_normalization_5_3332:`(
batch_normalization_5_3334:`(
batch_normalization_5_3336:`!
tf_image_resize_2_resize_size'
conv2d_6_3344:p0
conv2d_6_3346:0(
batch_normalization_6_3349:0(
batch_normalization_6_3351:0(
batch_normalization_6_3353:0(
batch_normalization_6_3355:0!
tf_image_resize_3_resize_size'
conv2d_7_3361:0
conv2d_7_3363:
identityИҐ+batch_normalization/StatefulPartitionedCallҐ-batch_normalization_1/StatefulPartitionedCallҐ-batch_normalization_2/StatefulPartitionedCallҐ-batch_normalization_3/StatefulPartitionedCallҐ-batch_normalization_4/StatefulPartitionedCallҐ-batch_normalization_5/StatefulPartitionedCallҐ-batch_normalization_6/StatefulPartitionedCallҐconv2d/StatefulPartitionedCallҐ conv2d_1/StatefulPartitionedCallҐ conv2d_2/StatefulPartitionedCallҐ conv2d_3/StatefulPartitionedCallҐ conv2d_4/StatefulPartitionedCallҐ conv2d_5/StatefulPartitionedCallҐ conv2d_6/StatefulPartitionedCallҐ conv2d_7/StatefulPartitionedCallр
conv2d/StatefulPartitionedCallStatefulPartitionedCallinput_1conv2d_3238conv2d_3240*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:€€€€€€€€€АА*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *I
fDRB
@__inference_conv2d_layer_call_and_return_conditional_losses_1699ъ
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0batch_normalization_3243batch_normalization_3245batch_normalization_3247batch_normalization_3249*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:€€€€€€€€€АА*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *V
fQRO
M__inference_batch_normalization_layer_call_and_return_conditional_losses_2645л
re_lu/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:€€€€€€€€€АА* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *H
fCRA
?__inference_re_lu_layer_call_and_return_conditional_losses_1737г
max_pooling2d/PartitionedCallPartitionedCallre_lu/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€@@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_max_pooling2d_layer_call_and_return_conditional_losses_1743Х
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0conv2d_1_3254conv2d_1_3256*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€@@ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_conv2d_1_layer_call_and_return_conditional_losses_1755Ж
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0batch_normalization_1_3259batch_normalization_1_3261batch_normalization_1_3263batch_normalization_1_3265*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€@@ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *X
fSRQ
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_2580п
re_lu_1/PartitionedCallPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€@@ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_re_lu_1_layer_call_and_return_conditional_losses_1793й
max_pooling2d_1/PartitionedCallPartitionedCall re_lu_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€   * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *R
fMRK
I__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_1799Ч
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_1/PartitionedCall:output:0conv2d_2_3270conv2d_2_3272*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€  @*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_conv2d_2_layer_call_and_return_conditional_losses_1811Ж
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0batch_normalization_2_3275batch_normalization_2_3277batch_normalization_2_3279batch_normalization_2_3281*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€  @*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *X
fSRQ
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_2515п
re_lu_2/PartitionedCallPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€  @* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_re_lu_2_layer_call_and_return_conditional_losses_1849й
max_pooling2d_2/PartitionedCallPartitionedCall re_lu_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *R
fMRK
I__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_1855Ш
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_2/PartitionedCall:output:0conv2d_3_3286conv2d_3_3288*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_conv2d_3_layer_call_and_return_conditional_losses_1867З
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0batch_normalization_3_3291batch_normalization_3_3293batch_normalization_3_3295batch_normalization_3_3297*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *X
fSRQ
O__inference_batch_normalization_3_layer_call_and_return_conditional_losses_2450р
re_lu_3/PartitionedCallPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_re_lu_3_layer_call_and_return_conditional_losses_1905к
max_pooling2d_3/PartitionedCallPartitionedCall re_lu_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€А* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *R
fMRK
I__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_1911б
,tf.image.resize/resize/ResizeNearestNeighborResizeNearestNeighbor(max_pooling2d_3/PartitionedCall:output:0tf_image_resize_resize_size*
T0*0
_output_shapes
:€€€€€€€€€А*
half_pixel_centers(W
tf.concat/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :й
tf.concat/concatConcatV2=tf.image.resize/resize/ResizeNearestNeighbor:resized_images:0(max_pooling2d_2/PartitionedCall:output:0tf.concat/concat/axis:output:0*
N*
T0*0
_output_shapes
:€€€€€€€€€јЙ
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCalltf.concat/concat:output:0conv2d_4_3306conv2d_4_3308*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€ј*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_conv2d_4_layer_call_and_return_conditional_losses_1927З
-batch_normalization_4/StatefulPartitionedCallStatefulPartitionedCall)conv2d_4/StatefulPartitionedCall:output:0batch_normalization_4_3311batch_normalization_4_3313batch_normalization_4_3315batch_normalization_4_3317*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€ј*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *X
fSRQ
O__inference_batch_normalization_4_layer_call_and_return_conditional_losses_2385р
re_lu_4/PartitionedCallPartitionedCall6batch_normalization_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€ј* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_re_lu_4_layer_call_and_return_conditional_losses_1965Ё
.tf.image.resize_1/resize/ResizeNearestNeighborResizeNearestNeighbor re_lu_4/PartitionedCall:output:0tf_image_resize_1_resize_size*
T0*0
_output_shapes
:€€€€€€€€€  ј*
half_pixel_centers(Y
tf.concat_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :п
tf.concat_1/concatConcatV2?tf.image.resize_1/resize/ResizeNearestNeighbor:resized_images:0(max_pooling2d_1/PartitionedCall:output:0 tf.concat_1/concat/axis:output:0*
N*
T0*0
_output_shapes
:€€€€€€€€€  аК
 conv2d_5/StatefulPartitionedCallStatefulPartitionedCalltf.concat_1/concat:output:0conv2d_5_3325conv2d_5_3327*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€  `*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_conv2d_5_layer_call_and_return_conditional_losses_1981Ж
-batch_normalization_5/StatefulPartitionedCallStatefulPartitionedCall)conv2d_5/StatefulPartitionedCall:output:0batch_normalization_5_3330batch_normalization_5_3332batch_normalization_5_3334batch_normalization_5_3336*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€  `*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *X
fSRQ
O__inference_batch_normalization_5_layer_call_and_return_conditional_losses_2325п
re_lu_5/PartitionedCallPartitionedCall6batch_normalization_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€  `* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_re_lu_5_layer_call_and_return_conditional_losses_2019№
.tf.image.resize_2/resize/ResizeNearestNeighborResizeNearestNeighbor re_lu_5/PartitionedCall:output:0tf_image_resize_2_resize_size*
T0*/
_output_shapes
:€€€€€€€€€@@`*
half_pixel_centers(Y
tf.concat_2/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :м
tf.concat_2/concatConcatV2?tf.image.resize_2/resize/ResizeNearestNeighbor:resized_images:0&max_pooling2d/PartitionedCall:output:0 tf.concat_2/concat/axis:output:0*
N*
T0*/
_output_shapes
:€€€€€€€€€@@pК
 conv2d_6/StatefulPartitionedCallStatefulPartitionedCalltf.concat_2/concat:output:0conv2d_6_3344conv2d_6_3346*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€@@0*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_conv2d_6_layer_call_and_return_conditional_losses_2035Ж
-batch_normalization_6/StatefulPartitionedCallStatefulPartitionedCall)conv2d_6/StatefulPartitionedCall:output:0batch_normalization_6_3349batch_normalization_6_3351batch_normalization_6_3353batch_normalization_6_3355*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€@@0*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *X
fSRQ
O__inference_batch_normalization_6_layer_call_and_return_conditional_losses_2265п
re_lu_6/PartitionedCallPartitionedCall6batch_normalization_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€@@0* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_re_lu_6_layer_call_and_return_conditional_losses_2073ё
.tf.image.resize_3/resize/ResizeNearestNeighborResizeNearestNeighbor re_lu_6/PartitionedCall:output:0tf_image_resize_3_resize_size*
T0*1
_output_shapes
:€€€€€€€€€АА0*
half_pixel_centers(∞
 conv2d_7/StatefulPartitionedCallStatefulPartitionedCall?tf.image.resize_3/resize/ResizeNearestNeighbor:resized_images:0conv2d_7_3361conv2d_7_3363*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:€€€€€€€€€АА*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_conv2d_7_layer_call_and_return_conditional_losses_2088д
reshape/PartitionedCallPartitionedCall)conv2d_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:€€€€€€€€€АА* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_reshape_layer_call_and_return_conditional_losses_2108y
IdentityIdentity reshape/PartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:€€€€€€€€€АА™
NoOpNoOp,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall.^batch_normalization_3/StatefulPartitionedCall.^batch_normalization_4/StatefulPartitionedCall.^batch_normalization_5/StatefulPartitionedCall.^batch_normalization_6/StatefulPartitionedCall^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall!^conv2d_4/StatefulPartitionedCall!^conv2d_5/StatefulPartitionedCall!^conv2d_6/StatefulPartitionedCall!^conv2d_7/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Ґ
_input_shapesР
Н:€€€€€€€€€АА: : : : : : : : : : : : : : : : : : : : : : : : :: : : : : : :: : : : : : :: : : : : : :: : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2^
-batch_normalization_3/StatefulPartitionedCall-batch_normalization_3/StatefulPartitionedCall2^
-batch_normalization_4/StatefulPartitionedCall-batch_normalization_4/StatefulPartitionedCall2^
-batch_normalization_5/StatefulPartitionedCall-batch_normalization_5/StatefulPartitionedCall2^
-batch_normalization_6/StatefulPartitionedCall-batch_normalization_6/StatefulPartitionedCall2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall2D
 conv2d_5/StatefulPartitionedCall conv2d_5/StatefulPartitionedCall2D
 conv2d_6/StatefulPartitionedCall conv2d_6/StatefulPartitionedCall2D
 conv2d_7/StatefulPartitionedCall conv2d_7/StatefulPartitionedCall:Z V
1
_output_shapes
:€€€€€€€€€АА
!
_user_specified_name	input_1: 

_output_shapes
::  

_output_shapes
:: '

_output_shapes
:: .

_output_shapes
:
©

ь
B__inference_conv2d_5_layer_call_and_return_conditional_losses_1981

inputs9
conv2d_readvariableop_resource:а`-
biasadd_readvariableop_resource:`
identityИҐBiasAdd/ReadVariableOpҐConv2D/ReadVariableOp}
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:а`*
dtype0Щ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€  `*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:`*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€  `g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:€€€€€€€€€  `w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :€€€€€€€€€  а: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:€€€€€€€€€  а
 
_user_specified_nameinputs
®
e
I__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_4737

inputs
identityЗ
MaxPoolMaxPoolinputs*0
_output_shapes
:€€€€€€€€€А*
ksize
*
paddingSAME*
strides
a
IdentityIdentityMaxPool:output:0*
T0*0
_output_shapes
:€€€€€€€€€А"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:€€€€€€€€€А:X T
0
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
В
Ъ
O__inference_batch_normalization_5_layer_call_and_return_conditional_losses_5015

inputs%
readvariableop_resource:`'
readvariableop_1_resource:`6
(fusedbatchnormv3_readvariableop_resource:`8
*fusedbatchnormv3_readvariableop_1_resource:`
identityИҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:`*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:`*
dtype0Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:`*
dtype0И
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:`*
dtype0ґ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€  `:`:`:`:`:*
epsilon%oГ:*
is_training( k
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:€€€€€€€€€  `∞
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:€€€€€€€€€  `: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:€€€€€€€€€  `
 
_user_specified_nameinputs
Р	
ѕ
4__inference_batch_normalization_2_layer_call_fn_4423

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identityИҐStatefulPartitionedCallЫ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *X
fSRQ
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1360Й
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs
ѕ
”
4__inference_batch_normalization_4_layer_call_fn_4795

inputs
unknown:	ј
	unknown_0:	ј
	unknown_1:	ј
	unknown_2:	ј
identityИҐStatefulPartitionedCallК
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€ј*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *X
fSRQ
O__inference_batch_normalization_4_layer_call_and_return_conditional_losses_1950x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:€€€€€€€€€ј`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:€€€€€€€€€ј: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:€€€€€€€€€ј
 
_user_specified_nameinputs
уб
Ў'
E__inference_Mobile_Unet_layer_call_and_return_conditional_losses_3859

inputs?
%conv2d_conv2d_readvariableop_resource:4
&conv2d_biasadd_readvariableop_resource:9
+batch_normalization_readvariableop_resource:;
-batch_normalization_readvariableop_1_resource:J
<batch_normalization_fusedbatchnormv3_readvariableop_resource:L
>batch_normalization_fusedbatchnormv3_readvariableop_1_resource:A
'conv2d_1_conv2d_readvariableop_resource: 6
(conv2d_1_biasadd_readvariableop_resource: ;
-batch_normalization_1_readvariableop_resource: =
/batch_normalization_1_readvariableop_1_resource: L
>batch_normalization_1_fusedbatchnormv3_readvariableop_resource: N
@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource: A
'conv2d_2_conv2d_readvariableop_resource: @6
(conv2d_2_biasadd_readvariableop_resource:@;
-batch_normalization_2_readvariableop_resource:@=
/batch_normalization_2_readvariableop_1_resource:@L
>batch_normalization_2_fusedbatchnormv3_readvariableop_resource:@N
@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource:@B
'conv2d_3_conv2d_readvariableop_resource:@А7
(conv2d_3_biasadd_readvariableop_resource:	А<
-batch_normalization_3_readvariableop_resource:	А>
/batch_normalization_3_readvariableop_1_resource:	АM
>batch_normalization_3_fusedbatchnormv3_readvariableop_resource:	АO
@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource:	А
tf_image_resize_resize_sizeC
'conv2d_4_conv2d_readvariableop_resource:јј7
(conv2d_4_biasadd_readvariableop_resource:	ј<
-batch_normalization_4_readvariableop_resource:	ј>
/batch_normalization_4_readvariableop_1_resource:	јM
>batch_normalization_4_fusedbatchnormv3_readvariableop_resource:	јO
@batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource:	ј!
tf_image_resize_1_resize_sizeB
'conv2d_5_conv2d_readvariableop_resource:а`6
(conv2d_5_biasadd_readvariableop_resource:`;
-batch_normalization_5_readvariableop_resource:`=
/batch_normalization_5_readvariableop_1_resource:`L
>batch_normalization_5_fusedbatchnormv3_readvariableop_resource:`N
@batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource:`!
tf_image_resize_2_resize_sizeA
'conv2d_6_conv2d_readvariableop_resource:p06
(conv2d_6_biasadd_readvariableop_resource:0;
-batch_normalization_6_readvariableop_resource:0=
/batch_normalization_6_readvariableop_1_resource:0L
>batch_normalization_6_fusedbatchnormv3_readvariableop_resource:0N
@batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource:0!
tf_image_resize_3_resize_sizeA
'conv2d_7_conv2d_readvariableop_resource:06
(conv2d_7_biasadd_readvariableop_resource:
identityИҐ3batch_normalization/FusedBatchNormV3/ReadVariableOpҐ5batch_normalization/FusedBatchNormV3/ReadVariableOp_1Ґ"batch_normalization/ReadVariableOpҐ$batch_normalization/ReadVariableOp_1Ґ5batch_normalization_1/FusedBatchNormV3/ReadVariableOpҐ7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1Ґ$batch_normalization_1/ReadVariableOpҐ&batch_normalization_1/ReadVariableOp_1Ґ5batch_normalization_2/FusedBatchNormV3/ReadVariableOpҐ7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1Ґ$batch_normalization_2/ReadVariableOpҐ&batch_normalization_2/ReadVariableOp_1Ґ5batch_normalization_3/FusedBatchNormV3/ReadVariableOpҐ7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1Ґ$batch_normalization_3/ReadVariableOpҐ&batch_normalization_3/ReadVariableOp_1Ґ5batch_normalization_4/FusedBatchNormV3/ReadVariableOpҐ7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1Ґ$batch_normalization_4/ReadVariableOpҐ&batch_normalization_4/ReadVariableOp_1Ґ5batch_normalization_5/FusedBatchNormV3/ReadVariableOpҐ7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1Ґ$batch_normalization_5/ReadVariableOpҐ&batch_normalization_5/ReadVariableOp_1Ґ5batch_normalization_6/FusedBatchNormV3/ReadVariableOpҐ7batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1Ґ$batch_normalization_6/ReadVariableOpҐ&batch_normalization_6/ReadVariableOp_1Ґconv2d/BiasAdd/ReadVariableOpҐconv2d/Conv2D/ReadVariableOpҐconv2d_1/BiasAdd/ReadVariableOpҐconv2d_1/Conv2D/ReadVariableOpҐconv2d_2/BiasAdd/ReadVariableOpҐconv2d_2/Conv2D/ReadVariableOpҐconv2d_3/BiasAdd/ReadVariableOpҐconv2d_3/Conv2D/ReadVariableOpҐconv2d_4/BiasAdd/ReadVariableOpҐconv2d_4/Conv2D/ReadVariableOpҐconv2d_5/BiasAdd/ReadVariableOpҐconv2d_5/Conv2D/ReadVariableOpҐconv2d_6/BiasAdd/ReadVariableOpҐconv2d_6/Conv2D/ReadVariableOpҐconv2d_7/BiasAdd/ReadVariableOpҐconv2d_7/Conv2D/ReadVariableOpК
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0©
conv2d/Conv2DConv2Dinputs$conv2d/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:€€€€€€€€€АА*
paddingSAME*
strides
А
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ф
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:€€€€€€€€€ААК
"batch_normalization/ReadVariableOpReadVariableOp+batch_normalization_readvariableop_resource*
_output_shapes
:*
dtype0О
$batch_normalization/ReadVariableOp_1ReadVariableOp-batch_normalization_readvariableop_1_resource*
_output_shapes
:*
dtype0ђ
3batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOp<batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0∞
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp>batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0≠
$batch_normalization/FusedBatchNormV3FusedBatchNormV3conv2d/BiasAdd:output:0*batch_normalization/ReadVariableOp:value:0,batch_normalization/ReadVariableOp_1:value:0;batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0=batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:€€€€€€€€€АА:::::*
epsilon%oГ:*
is_training( x

re_lu/ReluRelu(batch_normalization/FusedBatchNormV3:y:0*
T0*1
_output_shapes
:€€€€€€€€€АА¶
max_pooling2d/MaxPoolMaxPoolre_lu/Relu:activations:0*/
_output_shapes
:€€€€€€€€€@@*
ksize
*
paddingSAME*
strides
О
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0√
conv2d_1/Conv2DConv2Dmax_pooling2d/MaxPool:output:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€@@ *
paddingSAME*
strides
Д
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0Ш
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€@@ О
$batch_normalization_1/ReadVariableOpReadVariableOp-batch_normalization_1_readvariableop_resource*
_output_shapes
: *
dtype0Т
&batch_normalization_1/ReadVariableOp_1ReadVariableOp/batch_normalization_1_readvariableop_1_resource*
_output_shapes
: *
dtype0∞
5batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0і
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0Ј
&batch_normalization_1/FusedBatchNormV3FusedBatchNormV3conv2d_1/BiasAdd:output:0,batch_normalization_1/ReadVariableOp:value:0.batch_normalization_1/ReadVariableOp_1:value:0=batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€@@ : : : : :*
epsilon%oГ:*
is_training( z
re_lu_1/ReluRelu*batch_normalization_1/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:€€€€€€€€€@@ ™
max_pooling2d_1/MaxPoolMaxPoolre_lu_1/Relu:activations:0*/
_output_shapes
:€€€€€€€€€   *
ksize
*
paddingSAME*
strides
О
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0≈
conv2d_2/Conv2DConv2D max_pooling2d_1/MaxPool:output:0&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€  @*
paddingSAME*
strides
Д
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Ш
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€  @О
$batch_normalization_2/ReadVariableOpReadVariableOp-batch_normalization_2_readvariableop_resource*
_output_shapes
:@*
dtype0Т
&batch_normalization_2/ReadVariableOp_1ReadVariableOp/batch_normalization_2_readvariableop_1_resource*
_output_shapes
:@*
dtype0∞
5batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0і
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0Ј
&batch_normalization_2/FusedBatchNormV3FusedBatchNormV3conv2d_2/BiasAdd:output:0,batch_normalization_2/ReadVariableOp:value:0.batch_normalization_2/ReadVariableOp_1:value:0=batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€  @:@:@:@:@:*
epsilon%oГ:*
is_training( z
re_lu_2/ReluRelu*batch_normalization_2/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:€€€€€€€€€  @™
max_pooling2d_2/MaxPoolMaxPoolre_lu_2/Relu:activations:0*/
_output_shapes
:€€€€€€€€€@*
ksize
*
paddingSAME*
strides
П
conv2d_3/Conv2D/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*'
_output_shapes
:@А*
dtype0∆
conv2d_3/Conv2DConv2D max_pooling2d_2/MaxPool:output:0&conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€А*
paddingSAME*
strides
Е
conv2d_3/BiasAdd/ReadVariableOpReadVariableOp(conv2d_3_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0Щ
conv2d_3/BiasAddBiasAddconv2d_3/Conv2D:output:0'conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€АП
$batch_normalization_3/ReadVariableOpReadVariableOp-batch_normalization_3_readvariableop_resource*
_output_shapes	
:А*
dtype0У
&batch_normalization_3/ReadVariableOp_1ReadVariableOp/batch_normalization_3_readvariableop_1_resource*
_output_shapes	
:А*
dtype0±
5batch_normalization_3/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype0µ
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype0Љ
&batch_normalization_3/FusedBatchNormV3FusedBatchNormV3conv2d_3/BiasAdd:output:0,batch_normalization_3/ReadVariableOp:value:0.batch_normalization_3/ReadVariableOp_1:value:0=batch_normalization_3/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:€€€€€€€€€А:А:А:А:А:*
epsilon%oГ:*
is_training( {
re_lu_3/ReluRelu*batch_normalization_3/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:€€€€€€€€€АЂ
max_pooling2d_3/MaxPoolMaxPoolre_lu_3/Relu:activations:0*0
_output_shapes
:€€€€€€€€€А*
ksize
*
paddingSAME*
strides
ў
,tf.image.resize/resize/ResizeNearestNeighborResizeNearestNeighbor max_pooling2d_3/MaxPool:output:0tf_image_resize_resize_size*
T0*0
_output_shapes
:€€€€€€€€€А*
half_pixel_centers(W
tf.concat/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :б
tf.concat/concatConcatV2=tf.image.resize/resize/ResizeNearestNeighbor:resized_images:0 max_pooling2d_2/MaxPool:output:0tf.concat/concat/axis:output:0*
N*
T0*0
_output_shapes
:€€€€€€€€€јР
conv2d_4/Conv2D/ReadVariableOpReadVariableOp'conv2d_4_conv2d_readvariableop_resource*(
_output_shapes
:јј*
dtype0њ
conv2d_4/Conv2DConv2Dtf.concat/concat:output:0&conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€ј*
paddingSAME*
strides
Е
conv2d_4/BiasAdd/ReadVariableOpReadVariableOp(conv2d_4_biasadd_readvariableop_resource*
_output_shapes	
:ј*
dtype0Щ
conv2d_4/BiasAddBiasAddconv2d_4/Conv2D:output:0'conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€јП
$batch_normalization_4/ReadVariableOpReadVariableOp-batch_normalization_4_readvariableop_resource*
_output_shapes	
:ј*
dtype0У
&batch_normalization_4/ReadVariableOp_1ReadVariableOp/batch_normalization_4_readvariableop_1_resource*
_output_shapes	
:ј*
dtype0±
5batch_normalization_4/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_4_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:ј*
dtype0µ
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:ј*
dtype0Љ
&batch_normalization_4/FusedBatchNormV3FusedBatchNormV3conv2d_4/BiasAdd:output:0,batch_normalization_4/ReadVariableOp:value:0.batch_normalization_4/ReadVariableOp_1:value:0=batch_normalization_4/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:€€€€€€€€€ј:ј:ј:ј:ј:*
epsilon%oГ:*
is_training( {
re_lu_4/ReluRelu*batch_normalization_4/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:€€€€€€€€€ј„
.tf.image.resize_1/resize/ResizeNearestNeighborResizeNearestNeighborre_lu_4/Relu:activations:0tf_image_resize_1_resize_size*
T0*0
_output_shapes
:€€€€€€€€€  ј*
half_pixel_centers(Y
tf.concat_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :з
tf.concat_1/concatConcatV2?tf.image.resize_1/resize/ResizeNearestNeighbor:resized_images:0 max_pooling2d_1/MaxPool:output:0 tf.concat_1/concat/axis:output:0*
N*
T0*0
_output_shapes
:€€€€€€€€€  аП
conv2d_5/Conv2D/ReadVariableOpReadVariableOp'conv2d_5_conv2d_readvariableop_resource*'
_output_shapes
:а`*
dtype0ј
conv2d_5/Conv2DConv2Dtf.concat_1/concat:output:0&conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€  `*
paddingSAME*
strides
Д
conv2d_5/BiasAdd/ReadVariableOpReadVariableOp(conv2d_5_biasadd_readvariableop_resource*
_output_shapes
:`*
dtype0Ш
conv2d_5/BiasAddBiasAddconv2d_5/Conv2D:output:0'conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€  `О
$batch_normalization_5/ReadVariableOpReadVariableOp-batch_normalization_5_readvariableop_resource*
_output_shapes
:`*
dtype0Т
&batch_normalization_5/ReadVariableOp_1ReadVariableOp/batch_normalization_5_readvariableop_1_resource*
_output_shapes
:`*
dtype0∞
5batch_normalization_5/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_5_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:`*
dtype0і
7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:`*
dtype0Ј
&batch_normalization_5/FusedBatchNormV3FusedBatchNormV3conv2d_5/BiasAdd:output:0,batch_normalization_5/ReadVariableOp:value:0.batch_normalization_5/ReadVariableOp_1:value:0=batch_normalization_5/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€  `:`:`:`:`:*
epsilon%oГ:*
is_training( z
re_lu_5/ReluRelu*batch_normalization_5/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:€€€€€€€€€  `÷
.tf.image.resize_2/resize/ResizeNearestNeighborResizeNearestNeighborre_lu_5/Relu:activations:0tf_image_resize_2_resize_size*
T0*/
_output_shapes
:€€€€€€€€€@@`*
half_pixel_centers(Y
tf.concat_2/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :д
tf.concat_2/concatConcatV2?tf.image.resize_2/resize/ResizeNearestNeighbor:resized_images:0max_pooling2d/MaxPool:output:0 tf.concat_2/concat/axis:output:0*
N*
T0*/
_output_shapes
:€€€€€€€€€@@pО
conv2d_6/Conv2D/ReadVariableOpReadVariableOp'conv2d_6_conv2d_readvariableop_resource*&
_output_shapes
:p0*
dtype0ј
conv2d_6/Conv2DConv2Dtf.concat_2/concat:output:0&conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€@@0*
paddingSAME*
strides
Д
conv2d_6/BiasAdd/ReadVariableOpReadVariableOp(conv2d_6_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype0Ш
conv2d_6/BiasAddBiasAddconv2d_6/Conv2D:output:0'conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€@@0О
$batch_normalization_6/ReadVariableOpReadVariableOp-batch_normalization_6_readvariableop_resource*
_output_shapes
:0*
dtype0Т
&batch_normalization_6/ReadVariableOp_1ReadVariableOp/batch_normalization_6_readvariableop_1_resource*
_output_shapes
:0*
dtype0∞
5batch_normalization_6/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_6_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype0і
7batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype0Ј
&batch_normalization_6/FusedBatchNormV3FusedBatchNormV3conv2d_6/BiasAdd:output:0,batch_normalization_6/ReadVariableOp:value:0.batch_normalization_6/ReadVariableOp_1:value:0=batch_normalization_6/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€@@0:0:0:0:0:*
epsilon%oГ:*
is_training( z
re_lu_6/ReluRelu*batch_normalization_6/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:€€€€€€€€€@@0Ў
.tf.image.resize_3/resize/ResizeNearestNeighborResizeNearestNeighborre_lu_6/Relu:activations:0tf_image_resize_3_resize_size*
T0*1
_output_shapes
:€€€€€€€€€АА0*
half_pixel_centers(О
conv2d_7/Conv2D/ReadVariableOpReadVariableOp'conv2d_7_conv2d_readvariableop_resource*&
_output_shapes
:0*
dtype0ж
conv2d_7/Conv2DConv2D?tf.image.resize_3/resize/ResizeNearestNeighbor:resized_images:0&conv2d_7/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:€€€€€€€€€АА*
paddingSAME*
strides
Д
conv2d_7/BiasAdd/ReadVariableOpReadVariableOp(conv2d_7_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ъ
conv2d_7/BiasAddBiasAddconv2d_7/Conv2D:output:0'conv2d_7/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:€€€€€€€€€ААr
conv2d_7/SigmoidSigmoidconv2d_7/BiasAdd:output:0*
T0*1
_output_shapes
:€€€€€€€€€ААQ
reshape/ShapeShapeconv2d_7/Sigmoid:y:0*
T0*
_output_shapes
:e
reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: g
reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:g
reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:щ
reshape/strided_sliceStridedSlicereshape/Shape:output:0$reshape/strided_slice/stack:output:0&reshape/strided_slice/stack_1:output:0&reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskY
reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :Z
reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value
B :АZ
reshape/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value
B :А—
reshape/Reshape/shapePackreshape/strided_slice:output:0 reshape/Reshape/shape/1:output:0 reshape/Reshape/shape/2:output:0 reshape/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:М
reshape/ReshapeReshapeconv2d_7/Sigmoid:y:0reshape/Reshape/shape:output:0*
T0*1
_output_shapes
:€€€€€€€€€ААq
IdentityIdentityreshape/Reshape:output:0^NoOp*
T0*1
_output_shapes
:€€€€€€€€€АА†
NoOpNoOp4^batch_normalization/FusedBatchNormV3/ReadVariableOp6^batch_normalization/FusedBatchNormV3/ReadVariableOp_1#^batch_normalization/ReadVariableOp%^batch_normalization/ReadVariableOp_16^batch_normalization_1/FusedBatchNormV3/ReadVariableOp8^batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_1/ReadVariableOp'^batch_normalization_1/ReadVariableOp_16^batch_normalization_2/FusedBatchNormV3/ReadVariableOp8^batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_2/ReadVariableOp'^batch_normalization_2/ReadVariableOp_16^batch_normalization_3/FusedBatchNormV3/ReadVariableOp8^batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_3/ReadVariableOp'^batch_normalization_3/ReadVariableOp_16^batch_normalization_4/FusedBatchNormV3/ReadVariableOp8^batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_4/ReadVariableOp'^batch_normalization_4/ReadVariableOp_16^batch_normalization_5/FusedBatchNormV3/ReadVariableOp8^batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_5/ReadVariableOp'^batch_normalization_5/ReadVariableOp_16^batch_normalization_6/FusedBatchNormV3/ReadVariableOp8^batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_6/ReadVariableOp'^batch_normalization_6/ReadVariableOp_1^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp ^conv2d_3/BiasAdd/ReadVariableOp^conv2d_3/Conv2D/ReadVariableOp ^conv2d_4/BiasAdd/ReadVariableOp^conv2d_4/Conv2D/ReadVariableOp ^conv2d_5/BiasAdd/ReadVariableOp^conv2d_5/Conv2D/ReadVariableOp ^conv2d_6/BiasAdd/ReadVariableOp^conv2d_6/Conv2D/ReadVariableOp ^conv2d_7/BiasAdd/ReadVariableOp^conv2d_7/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Ґ
_input_shapesР
Н:€€€€€€€€€АА: : : : : : : : : : : : : : : : : : : : : : : : :: : : : : : :: : : : : : :: : : : : : :: : 2j
3batch_normalization/FusedBatchNormV3/ReadVariableOp3batch_normalization/FusedBatchNormV3/ReadVariableOp2n
5batch_normalization/FusedBatchNormV3/ReadVariableOp_15batch_normalization/FusedBatchNormV3/ReadVariableOp_12H
"batch_normalization/ReadVariableOp"batch_normalization/ReadVariableOp2L
$batch_normalization/ReadVariableOp_1$batch_normalization/ReadVariableOp_12n
5batch_normalization_1/FusedBatchNormV3/ReadVariableOp5batch_normalization_1/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_17batch_normalization_1/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_1/ReadVariableOp$batch_normalization_1/ReadVariableOp2P
&batch_normalization_1/ReadVariableOp_1&batch_normalization_1/ReadVariableOp_12n
5batch_normalization_2/FusedBatchNormV3/ReadVariableOp5batch_normalization_2/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_17batch_normalization_2/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_2/ReadVariableOp$batch_normalization_2/ReadVariableOp2P
&batch_normalization_2/ReadVariableOp_1&batch_normalization_2/ReadVariableOp_12n
5batch_normalization_3/FusedBatchNormV3/ReadVariableOp5batch_normalization_3/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_17batch_normalization_3/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_3/ReadVariableOp$batch_normalization_3/ReadVariableOp2P
&batch_normalization_3/ReadVariableOp_1&batch_normalization_3/ReadVariableOp_12n
5batch_normalization_4/FusedBatchNormV3/ReadVariableOp5batch_normalization_4/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_17batch_normalization_4/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_4/ReadVariableOp$batch_normalization_4/ReadVariableOp2P
&batch_normalization_4/ReadVariableOp_1&batch_normalization_4/ReadVariableOp_12n
5batch_normalization_5/FusedBatchNormV3/ReadVariableOp5batch_normalization_5/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_17batch_normalization_5/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_5/ReadVariableOp$batch_normalization_5/ReadVariableOp2P
&batch_normalization_5/ReadVariableOp_1&batch_normalization_5/ReadVariableOp_12n
5batch_normalization_6/FusedBatchNormV3/ReadVariableOp5batch_normalization_6/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_6/FusedBatchNormV3/ReadVariableOp_17batch_normalization_6/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_6/ReadVariableOp$batch_normalization_6/ReadVariableOp2P
&batch_normalization_6/ReadVariableOp_1&batch_normalization_6/ReadVariableOp_12>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp2B
conv2d_2/BiasAdd/ReadVariableOpconv2d_2/BiasAdd/ReadVariableOp2@
conv2d_2/Conv2D/ReadVariableOpconv2d_2/Conv2D/ReadVariableOp2B
conv2d_3/BiasAdd/ReadVariableOpconv2d_3/BiasAdd/ReadVariableOp2@
conv2d_3/Conv2D/ReadVariableOpconv2d_3/Conv2D/ReadVariableOp2B
conv2d_4/BiasAdd/ReadVariableOpconv2d_4/BiasAdd/ReadVariableOp2@
conv2d_4/Conv2D/ReadVariableOpconv2d_4/Conv2D/ReadVariableOp2B
conv2d_5/BiasAdd/ReadVariableOpconv2d_5/BiasAdd/ReadVariableOp2@
conv2d_5/Conv2D/ReadVariableOpconv2d_5/Conv2D/ReadVariableOp2B
conv2d_6/BiasAdd/ReadVariableOpconv2d_6/BiasAdd/ReadVariableOp2@
conv2d_6/Conv2D/ReadVariableOpconv2d_6/Conv2D/ReadVariableOp2B
conv2d_7/BiasAdd/ReadVariableOpconv2d_7/BiasAdd/ReadVariableOp2@
conv2d_7/Conv2D/ReadVariableOpconv2d_7/Conv2D/ReadVariableOp:Y U
1
_output_shapes
:€€€€€€€€€АА
 
_user_specified_nameinputs: 

_output_shapes
::  

_output_shapes
:: '

_output_shapes
:: .

_output_shapes
:
 
Ъ
O__inference_batch_normalization_6_layer_call_and_return_conditional_losses_5132

inputs%
readvariableop_resource:0'
readvariableop_1_resource:06
(fusedbatchnormv3_readvariableop_resource:08
*fusedbatchnormv3_readvariableop_1_resource:0
identityИҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:0*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:0*
dtype0Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype0И
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype0»
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€0:0:0:0:0:*
epsilon%oГ:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€0∞
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+€€€€€€€€€€€€€€€€€€€€€€€€€€€0: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€0
 
_user_specified_nameinputs
О
c
G__inference_max_pooling2d_layer_call_and_return_conditional_losses_1259

inputs
identity°
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€*
ksize
*
paddingSAME*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:r n
J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
≈
ѕ
4__inference_batch_normalization_2_layer_call_fn_4462

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identityИҐStatefulPartitionedCallЗ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€  @*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *X
fSRQ
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_2515w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:€€€€€€€€€  @`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:€€€€€€€€€  @: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:€€€€€€€€€  @
 
_user_specified_nameinputs
Ў
Њ
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_4498

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identityИҐAssignNewValueҐAssignNewValue_1ҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0И
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0÷
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@:@:@:@:@:*
epsilon%oГ:*
exponential_avg_factor%
„#<∞
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0Ї
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@‘
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs
е
]
A__inference_re_lu_6_layer_call_and_return_conditional_losses_5196

inputs
identityN
ReluReluinputs*
T0*/
_output_shapes
:€€€€€€€€€@@0b
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:€€€€€€€€€@@0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:€€€€€€€€€@@0:W S
/
_output_shapes
:€€€€€€€€€@@0
 
_user_specified_nameinputs
е
]
A__inference_re_lu_6_layer_call_and_return_conditional_losses_2073

inputs
identityN
ReluReluinputs*
T0*/
_output_shapes
:€€€€€€€€€@@0b
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:€€€€€€€€€@@0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:€€€€€€€€€@@0:W S
/
_output_shapes
:€€€€€€€€€@@0
 
_user_specified_nameinputs
н
Ю
'__inference_conv2d_3_layer_call_fn_4573

inputs"
unknown:@А
	unknown_0:	А
identityИҐStatefulPartitionedCallг
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_conv2d_3_layer_call_and_return_conditional_losses_1867x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:€€€€€€€€€А`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:€€€€€€€€€@: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:€€€€€€€€€@
 
_user_specified_nameinputs
«
ѕ
4__inference_batch_normalization_6_layer_call_fn_5101

inputs
unknown:0
	unknown_0:0
	unknown_1:0
	unknown_2:0
identityИҐStatefulPartitionedCallЙ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€@@0*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *X
fSRQ
O__inference_batch_normalization_6_layer_call_and_return_conditional_losses_2058w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:€€€€€€€€€@@0`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:€€€€€€€€€@@0: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:€€€€€€€€€@@0
 
_user_specified_nameinputs
В
Ъ
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1834

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identityИҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0И
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0ґ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€  @:@:@:@:@:*
epsilon%oГ:*
is_training( k
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:€€€€€€€€€  @∞
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:€€€€€€€€€  @: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:€€€€€€€€€  @
 
_user_specified_nameinputs
Р
Њ
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_4361

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identityИҐAssignNewValueҐAssignNewValue_1ҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0И
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0ƒ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:€€€€€€€€€@@ : : : : :*
epsilon%oГ:*
exponential_avg_factor%
„#<∞
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0Ї
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0k
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:€€€€€€€€€@@ ‘
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:€€€€€€€€€@@ : : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:€€€€€€€€€@@ 
 
_user_specified_nameinputs
й
Ь
'__inference_conv2d_6_layer_call_fn_5052

inputs!
unknown:p0
	unknown_0:0
identityИҐStatefulPartitionedCallв
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€@@0*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_conv2d_6_layer_call_and_return_conditional_losses_2035w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:€€€€€€€€€@@0`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:€€€€€€€€€@@p: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:€€€€€€€€€@@p
 
_user_specified_nameinputs
Р
e
I__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_4386

inputs
identity°
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€*
ksize
*
paddingSAME*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:r n
J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
Т
Ю
O__inference_batch_normalization_4_layer_call_and_return_conditional_losses_1950

inputs&
readvariableop_resource:	ј(
readvariableop_1_resource:	ј7
(fusedbatchnormv3_readvariableop_resource:	ј9
*fusedbatchnormv3_readvariableop_1_resource:	ј
identityИҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:ј*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:ј*
dtype0Е
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:ј*
dtype0Й
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:ј*
dtype0ї
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:€€€€€€€€€ј:ј:ј:ј:ј:*
epsilon%oГ:*
is_training( l
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*0
_output_shapes
:€€€€€€€€€ј∞
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:€€€€€€€€€ј: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:X T
0
_output_shapes
:€€€€€€€€€ј
 
_user_specified_nameinputs
О	
ѕ
4__inference_batch_normalization_2_layer_call_fn_4436

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identityИҐStatefulPartitionedCallЩ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *X
fSRQ
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1391Й
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
 
_user_specified_nameinputs
О	
ѕ
4__inference_batch_normalization_6_layer_call_fn_5088

inputs
unknown:0
	unknown_0:0
	unknown_1:0
	unknown_2:0
identityИҐStatefulPartitionedCallЩ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€0*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *X
fSRQ
O__inference_batch_normalization_6_layer_call_and_return_conditional_losses_1671Й
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€0`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+€€€€€€€€€€€€€€€€€€€€€€€€€€€0: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€0
 
_user_specified_nameinputs
≠

щ
@__inference_conv2d_layer_call_and_return_conditional_losses_4064

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identityИҐBiasAdd/ReadVariableOpҐConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0Ы
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:€€€€€€€€€АА*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:€€€€€€€€€ААi
IdentityIdentityBiasAdd:output:0^NoOp*
T0*1
_output_shapes
:€€€€€€€€€ААw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:€€€€€€€€€АА: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:€€€€€€€€€АА
 
_user_specified_nameinputs
•

ы
B__inference_conv2d_1_layer_call_and_return_conditional_losses_4237

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identityИҐBiasAdd/ReadVariableOpҐConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0Щ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€@@ *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€@@ g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:€€€€€€€€€@@ w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:€€€€€€€€€@@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:€€€€€€€€€@@
 
_user_specified_nameinputs
Р
e
I__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_1335

inputs
identity°
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€*
ksize
*
paddingSAME*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:r n
J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
Ј
J
.__inference_max_pooling2d_1_layer_call_fn_4376

inputs
identityЏ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *R
fMRK
I__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_1335Г
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:r n
J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
Ч
х

"__inference_signature_wrapper_3471
input_1!
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:#
	unknown_5: 
	unknown_6: 
	unknown_7: 
	unknown_8: 
	unknown_9: 

unknown_10: $

unknown_11: @

unknown_12:@

unknown_13:@

unknown_14:@

unknown_15:@

unknown_16:@%

unknown_17:@А

unknown_18:	А

unknown_19:	А

unknown_20:	А

unknown_21:	А

unknown_22:	А

unknown_23&

unknown_24:јј

unknown_25:	ј

unknown_26:	ј

unknown_27:	ј

unknown_28:	ј

unknown_29:	ј

unknown_30%

unknown_31:а`

unknown_32:`

unknown_33:`

unknown_34:`

unknown_35:`

unknown_36:`

unknown_37$

unknown_38:p0

unknown_39:0

unknown_40:0

unknown_41:0

unknown_42:0

unknown_43:0

unknown_44$

unknown_45:0

unknown_46:
identityИҐStatefulPartitionedCallє
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46*<
Tin5
321*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:€€€€€€€€€АА*N
_read_only_resource_inputs0
.,	
!"#$%&()*+,-/0*0
config_proto 

CPU

GPU2*0J 8В *(
f#R!
__inference__wrapped_model_1186y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:€€€€€€€€€АА`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Ґ
_input_shapesР
Н:€€€€€€€€€АА: : : : : : : : : : : : : : : : : : : : : : : : :: : : : : : :: : : : : : :: : : : : : :: : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
1
_output_shapes
:€€€€€€€€€АА
!
_user_specified_name	input_1: 

_output_shapes
::  

_output_shapes
:: '

_output_shapes
:: .

_output_shapes
:
Џ
Ю
O__inference_batch_normalization_3_layer_call_and_return_conditional_losses_1436

inputs&
readvariableop_resource:	А(
readvariableop_1_resource:	А7
(fusedbatchnormv3_readvariableop_resource:	А9
*fusedbatchnormv3_readvariableop_1_resource:	А
identityИҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:А*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:А*
dtype0Е
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:А*
dtype0Й
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:А*
dtype0Ќ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А:А:А:А:А:*
epsilon%oГ:*
is_training( ~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А∞
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
 
_user_specified_nameinputs
К	
Ќ
2__inference_batch_normalization_layer_call_fn_4090

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identityИҐStatefulPartitionedCallЧ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *V
fQRO
M__inference_batch_normalization_layer_call_and_return_conditional_losses_1239Й
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+€€€€€€€€€€€€€€€€€€€€€€€€€€€: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
©§
…D
 __inference__traced_restore_5896
file_prefix8
assignvariableop_conv2d_kernel:,
assignvariableop_1_conv2d_bias::
,assignvariableop_2_batch_normalization_gamma:9
+assignvariableop_3_batch_normalization_beta:@
2assignvariableop_4_batch_normalization_moving_mean:D
6assignvariableop_5_batch_normalization_moving_variance:<
"assignvariableop_6_conv2d_1_kernel: .
 assignvariableop_7_conv2d_1_bias: <
.assignvariableop_8_batch_normalization_1_gamma: ;
-assignvariableop_9_batch_normalization_1_beta: C
5assignvariableop_10_batch_normalization_1_moving_mean: G
9assignvariableop_11_batch_normalization_1_moving_variance: =
#assignvariableop_12_conv2d_2_kernel: @/
!assignvariableop_13_conv2d_2_bias:@=
/assignvariableop_14_batch_normalization_2_gamma:@<
.assignvariableop_15_batch_normalization_2_beta:@C
5assignvariableop_16_batch_normalization_2_moving_mean:@G
9assignvariableop_17_batch_normalization_2_moving_variance:@>
#assignvariableop_18_conv2d_3_kernel:@А0
!assignvariableop_19_conv2d_3_bias:	А>
/assignvariableop_20_batch_normalization_3_gamma:	А=
.assignvariableop_21_batch_normalization_3_beta:	АD
5assignvariableop_22_batch_normalization_3_moving_mean:	АH
9assignvariableop_23_batch_normalization_3_moving_variance:	А?
#assignvariableop_24_conv2d_4_kernel:јј0
!assignvariableop_25_conv2d_4_bias:	ј>
/assignvariableop_26_batch_normalization_4_gamma:	ј=
.assignvariableop_27_batch_normalization_4_beta:	јD
5assignvariableop_28_batch_normalization_4_moving_mean:	јH
9assignvariableop_29_batch_normalization_4_moving_variance:	ј>
#assignvariableop_30_conv2d_5_kernel:а`/
!assignvariableop_31_conv2d_5_bias:`=
/assignvariableop_32_batch_normalization_5_gamma:`<
.assignvariableop_33_batch_normalization_5_beta:`C
5assignvariableop_34_batch_normalization_5_moving_mean:`G
9assignvariableop_35_batch_normalization_5_moving_variance:`=
#assignvariableop_36_conv2d_6_kernel:p0/
!assignvariableop_37_conv2d_6_bias:0=
/assignvariableop_38_batch_normalization_6_gamma:0<
.assignvariableop_39_batch_normalization_6_beta:0C
5assignvariableop_40_batch_normalization_6_moving_mean:0G
9assignvariableop_41_batch_normalization_6_moving_variance:0=
#assignvariableop_42_conv2d_7_kernel:0/
!assignvariableop_43_conv2d_7_bias:=
#assignvariableop_44_conv2d_kernel_m:/
!assignvariableop_45_conv2d_bias_m:=
/assignvariableop_46_batch_normalization_gamma_m:<
.assignvariableop_47_batch_normalization_beta_m:?
%assignvariableop_48_conv2d_1_kernel_m: 1
#assignvariableop_49_conv2d_1_bias_m: ?
1assignvariableop_50_batch_normalization_1_gamma_m: >
0assignvariableop_51_batch_normalization_1_beta_m: ?
%assignvariableop_52_conv2d_2_kernel_m: @1
#assignvariableop_53_conv2d_2_bias_m:@?
1assignvariableop_54_batch_normalization_2_gamma_m:@>
0assignvariableop_55_batch_normalization_2_beta_m:@@
%assignvariableop_56_conv2d_3_kernel_m:@А2
#assignvariableop_57_conv2d_3_bias_m:	А@
1assignvariableop_58_batch_normalization_3_gamma_m:	А?
0assignvariableop_59_batch_normalization_3_beta_m:	АA
%assignvariableop_60_conv2d_4_kernel_m:јј2
#assignvariableop_61_conv2d_4_bias_m:	ј@
1assignvariableop_62_batch_normalization_4_gamma_m:	ј?
0assignvariableop_63_batch_normalization_4_beta_m:	ј@
%assignvariableop_64_conv2d_5_kernel_m:а`1
#assignvariableop_65_conv2d_5_bias_m:`?
1assignvariableop_66_batch_normalization_5_gamma_m:`>
0assignvariableop_67_batch_normalization_5_beta_m:`?
%assignvariableop_68_conv2d_6_kernel_m:p01
#assignvariableop_69_conv2d_6_bias_m:0?
1assignvariableop_70_batch_normalization_6_gamma_m:0>
0assignvariableop_71_batch_normalization_6_beta_m:0?
%assignvariableop_72_conv2d_7_kernel_m:01
#assignvariableop_73_conv2d_7_bias_m:=
#assignvariableop_74_conv2d_kernel_v:/
!assignvariableop_75_conv2d_bias_v:=
/assignvariableop_76_batch_normalization_gamma_v:<
.assignvariableop_77_batch_normalization_beta_v:?
%assignvariableop_78_conv2d_1_kernel_v: 1
#assignvariableop_79_conv2d_1_bias_v: ?
1assignvariableop_80_batch_normalization_1_gamma_v: >
0assignvariableop_81_batch_normalization_1_beta_v: ?
%assignvariableop_82_conv2d_2_kernel_v: @1
#assignvariableop_83_conv2d_2_bias_v:@?
1assignvariableop_84_batch_normalization_2_gamma_v:@>
0assignvariableop_85_batch_normalization_2_beta_v:@@
%assignvariableop_86_conv2d_3_kernel_v:@А2
#assignvariableop_87_conv2d_3_bias_v:	А@
1assignvariableop_88_batch_normalization_3_gamma_v:	А?
0assignvariableop_89_batch_normalization_3_beta_v:	АA
%assignvariableop_90_conv2d_4_kernel_v:јј2
#assignvariableop_91_conv2d_4_bias_v:	ј@
1assignvariableop_92_batch_normalization_4_gamma_v:	ј?
0assignvariableop_93_batch_normalization_4_beta_v:	ј@
%assignvariableop_94_conv2d_5_kernel_v:а`1
#assignvariableop_95_conv2d_5_bias_v:`?
1assignvariableop_96_batch_normalization_5_gamma_v:`>
0assignvariableop_97_batch_normalization_5_beta_v:`?
%assignvariableop_98_conv2d_6_kernel_v:p01
#assignvariableop_99_conv2d_6_bias_v:0@
2assignvariableop_100_batch_normalization_6_gamma_v:0?
1assignvariableop_101_batch_normalization_6_beta_v:0@
&assignvariableop_102_conv2d_7_kernel_v:02
$assignvariableop_103_conv2d_7_bias_v:
identity_105ИҐAssignVariableOpҐAssignVariableOp_1ҐAssignVariableOp_10ҐAssignVariableOp_100ҐAssignVariableOp_101ҐAssignVariableOp_102ҐAssignVariableOp_103ҐAssignVariableOp_11ҐAssignVariableOp_12ҐAssignVariableOp_13ҐAssignVariableOp_14ҐAssignVariableOp_15ҐAssignVariableOp_16ҐAssignVariableOp_17ҐAssignVariableOp_18ҐAssignVariableOp_19ҐAssignVariableOp_2ҐAssignVariableOp_20ҐAssignVariableOp_21ҐAssignVariableOp_22ҐAssignVariableOp_23ҐAssignVariableOp_24ҐAssignVariableOp_25ҐAssignVariableOp_26ҐAssignVariableOp_27ҐAssignVariableOp_28ҐAssignVariableOp_29ҐAssignVariableOp_3ҐAssignVariableOp_30ҐAssignVariableOp_31ҐAssignVariableOp_32ҐAssignVariableOp_33ҐAssignVariableOp_34ҐAssignVariableOp_35ҐAssignVariableOp_36ҐAssignVariableOp_37ҐAssignVariableOp_38ҐAssignVariableOp_39ҐAssignVariableOp_4ҐAssignVariableOp_40ҐAssignVariableOp_41ҐAssignVariableOp_42ҐAssignVariableOp_43ҐAssignVariableOp_44ҐAssignVariableOp_45ҐAssignVariableOp_46ҐAssignVariableOp_47ҐAssignVariableOp_48ҐAssignVariableOp_49ҐAssignVariableOp_5ҐAssignVariableOp_50ҐAssignVariableOp_51ҐAssignVariableOp_52ҐAssignVariableOp_53ҐAssignVariableOp_54ҐAssignVariableOp_55ҐAssignVariableOp_56ҐAssignVariableOp_57ҐAssignVariableOp_58ҐAssignVariableOp_59ҐAssignVariableOp_6ҐAssignVariableOp_60ҐAssignVariableOp_61ҐAssignVariableOp_62ҐAssignVariableOp_63ҐAssignVariableOp_64ҐAssignVariableOp_65ҐAssignVariableOp_66ҐAssignVariableOp_67ҐAssignVariableOp_68ҐAssignVariableOp_69ҐAssignVariableOp_7ҐAssignVariableOp_70ҐAssignVariableOp_71ҐAssignVariableOp_72ҐAssignVariableOp_73ҐAssignVariableOp_74ҐAssignVariableOp_75ҐAssignVariableOp_76ҐAssignVariableOp_77ҐAssignVariableOp_78ҐAssignVariableOp_79ҐAssignVariableOp_8ҐAssignVariableOp_80ҐAssignVariableOp_81ҐAssignVariableOp_82ҐAssignVariableOp_83ҐAssignVariableOp_84ҐAssignVariableOp_85ҐAssignVariableOp_86ҐAssignVariableOp_87ҐAssignVariableOp_88ҐAssignVariableOp_89ҐAssignVariableOp_9ҐAssignVariableOp_90ҐAssignVariableOp_91ҐAssignVariableOp_92ҐAssignVariableOp_93ҐAssignVariableOp_94ҐAssignVariableOp_95ҐAssignVariableOp_96ҐAssignVariableOp_97ҐAssignVariableOp_98ҐAssignVariableOp_99т;
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:i*
dtype0*Ш;
valueО;BЛ;iB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-9/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-9/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-9/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-11/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-11/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-11/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-13/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-13/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-13/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-13/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-14/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-14/bias/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-9/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-11/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-13/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-13/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-14/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-14/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-9/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-11/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-13/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-13/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-14/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-14/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH≈
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:i*
dtype0*з
valueЁBЏiB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B Ѓ
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*Ї
_output_shapesІ
§:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*w
dtypesm
k2i[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:Й
AssignVariableOpAssignVariableOpassignvariableop_conv2d_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:Н
AssignVariableOp_1AssignVariableOpassignvariableop_1_conv2d_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:Ы
AssignVariableOp_2AssignVariableOp,assignvariableop_2_batch_normalization_gammaIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:Ъ
AssignVariableOp_3AssignVariableOp+assignvariableop_3_batch_normalization_betaIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:°
AssignVariableOp_4AssignVariableOp2assignvariableop_4_batch_normalization_moving_meanIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:•
AssignVariableOp_5AssignVariableOp6assignvariableop_5_batch_normalization_moving_varianceIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:С
AssignVariableOp_6AssignVariableOp"assignvariableop_6_conv2d_1_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:П
AssignVariableOp_7AssignVariableOp assignvariableop_7_conv2d_1_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:Э
AssignVariableOp_8AssignVariableOp.assignvariableop_8_batch_normalization_1_gammaIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:Ь
AssignVariableOp_9AssignVariableOp-assignvariableop_9_batch_normalization_1_betaIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:¶
AssignVariableOp_10AssignVariableOp5assignvariableop_10_batch_normalization_1_moving_meanIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:™
AssignVariableOp_11AssignVariableOp9assignvariableop_11_batch_normalization_1_moving_varianceIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:Ф
AssignVariableOp_12AssignVariableOp#assignvariableop_12_conv2d_2_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:Т
AssignVariableOp_13AssignVariableOp!assignvariableop_13_conv2d_2_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:†
AssignVariableOp_14AssignVariableOp/assignvariableop_14_batch_normalization_2_gammaIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:Я
AssignVariableOp_15AssignVariableOp.assignvariableop_15_batch_normalization_2_betaIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:¶
AssignVariableOp_16AssignVariableOp5assignvariableop_16_batch_normalization_2_moving_meanIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:™
AssignVariableOp_17AssignVariableOp9assignvariableop_17_batch_normalization_2_moving_varianceIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:Ф
AssignVariableOp_18AssignVariableOp#assignvariableop_18_conv2d_3_kernelIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:Т
AssignVariableOp_19AssignVariableOp!assignvariableop_19_conv2d_3_biasIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:†
AssignVariableOp_20AssignVariableOp/assignvariableop_20_batch_normalization_3_gammaIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:Я
AssignVariableOp_21AssignVariableOp.assignvariableop_21_batch_normalization_3_betaIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:¶
AssignVariableOp_22AssignVariableOp5assignvariableop_22_batch_normalization_3_moving_meanIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:™
AssignVariableOp_23AssignVariableOp9assignvariableop_23_batch_normalization_3_moving_varianceIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:Ф
AssignVariableOp_24AssignVariableOp#assignvariableop_24_conv2d_4_kernelIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:Т
AssignVariableOp_25AssignVariableOp!assignvariableop_25_conv2d_4_biasIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:†
AssignVariableOp_26AssignVariableOp/assignvariableop_26_batch_normalization_4_gammaIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:Я
AssignVariableOp_27AssignVariableOp.assignvariableop_27_batch_normalization_4_betaIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:¶
AssignVariableOp_28AssignVariableOp5assignvariableop_28_batch_normalization_4_moving_meanIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:™
AssignVariableOp_29AssignVariableOp9assignvariableop_29_batch_normalization_4_moving_varianceIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:Ф
AssignVariableOp_30AssignVariableOp#assignvariableop_30_conv2d_5_kernelIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:Т
AssignVariableOp_31AssignVariableOp!assignvariableop_31_conv2d_5_biasIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:†
AssignVariableOp_32AssignVariableOp/assignvariableop_32_batch_normalization_5_gammaIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:Я
AssignVariableOp_33AssignVariableOp.assignvariableop_33_batch_normalization_5_betaIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:¶
AssignVariableOp_34AssignVariableOp5assignvariableop_34_batch_normalization_5_moving_meanIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:™
AssignVariableOp_35AssignVariableOp9assignvariableop_35_batch_normalization_5_moving_varianceIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:Ф
AssignVariableOp_36AssignVariableOp#assignvariableop_36_conv2d_6_kernelIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:Т
AssignVariableOp_37AssignVariableOp!assignvariableop_37_conv2d_6_biasIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:†
AssignVariableOp_38AssignVariableOp/assignvariableop_38_batch_normalization_6_gammaIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:Я
AssignVariableOp_39AssignVariableOp.assignvariableop_39_batch_normalization_6_betaIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:¶
AssignVariableOp_40AssignVariableOp5assignvariableop_40_batch_normalization_6_moving_meanIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:™
AssignVariableOp_41AssignVariableOp9assignvariableop_41_batch_normalization_6_moving_varianceIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:Ф
AssignVariableOp_42AssignVariableOp#assignvariableop_42_conv2d_7_kernelIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:Т
AssignVariableOp_43AssignVariableOp!assignvariableop_43_conv2d_7_biasIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:Ф
AssignVariableOp_44AssignVariableOp#assignvariableop_44_conv2d_kernel_mIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:Т
AssignVariableOp_45AssignVariableOp!assignvariableop_45_conv2d_bias_mIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:†
AssignVariableOp_46AssignVariableOp/assignvariableop_46_batch_normalization_gamma_mIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:Я
AssignVariableOp_47AssignVariableOp.assignvariableop_47_batch_normalization_beta_mIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:Ц
AssignVariableOp_48AssignVariableOp%assignvariableop_48_conv2d_1_kernel_mIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:Ф
AssignVariableOp_49AssignVariableOp#assignvariableop_49_conv2d_1_bias_mIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:Ґ
AssignVariableOp_50AssignVariableOp1assignvariableop_50_batch_normalization_1_gamma_mIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:°
AssignVariableOp_51AssignVariableOp0assignvariableop_51_batch_normalization_1_beta_mIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:Ц
AssignVariableOp_52AssignVariableOp%assignvariableop_52_conv2d_2_kernel_mIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:Ф
AssignVariableOp_53AssignVariableOp#assignvariableop_53_conv2d_2_bias_mIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:Ґ
AssignVariableOp_54AssignVariableOp1assignvariableop_54_batch_normalization_2_gamma_mIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:°
AssignVariableOp_55AssignVariableOp0assignvariableop_55_batch_normalization_2_beta_mIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:Ц
AssignVariableOp_56AssignVariableOp%assignvariableop_56_conv2d_3_kernel_mIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:Ф
AssignVariableOp_57AssignVariableOp#assignvariableop_57_conv2d_3_bias_mIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:Ґ
AssignVariableOp_58AssignVariableOp1assignvariableop_58_batch_normalization_3_gamma_mIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:°
AssignVariableOp_59AssignVariableOp0assignvariableop_59_batch_normalization_3_beta_mIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:Ц
AssignVariableOp_60AssignVariableOp%assignvariableop_60_conv2d_4_kernel_mIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:Ф
AssignVariableOp_61AssignVariableOp#assignvariableop_61_conv2d_4_bias_mIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:Ґ
AssignVariableOp_62AssignVariableOp1assignvariableop_62_batch_normalization_4_gamma_mIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:°
AssignVariableOp_63AssignVariableOp0assignvariableop_63_batch_normalization_4_beta_mIdentity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:Ц
AssignVariableOp_64AssignVariableOp%assignvariableop_64_conv2d_5_kernel_mIdentity_64:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:Ф
AssignVariableOp_65AssignVariableOp#assignvariableop_65_conv2d_5_bias_mIdentity_65:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:Ґ
AssignVariableOp_66AssignVariableOp1assignvariableop_66_batch_normalization_5_gamma_mIdentity_66:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:°
AssignVariableOp_67AssignVariableOp0assignvariableop_67_batch_normalization_5_beta_mIdentity_67:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:Ц
AssignVariableOp_68AssignVariableOp%assignvariableop_68_conv2d_6_kernel_mIdentity_68:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:Ф
AssignVariableOp_69AssignVariableOp#assignvariableop_69_conv2d_6_bias_mIdentity_69:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_70IdentityRestoreV2:tensors:70"/device:CPU:0*
T0*
_output_shapes
:Ґ
AssignVariableOp_70AssignVariableOp1assignvariableop_70_batch_normalization_6_gamma_mIdentity_70:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_71IdentityRestoreV2:tensors:71"/device:CPU:0*
T0*
_output_shapes
:°
AssignVariableOp_71AssignVariableOp0assignvariableop_71_batch_normalization_6_beta_mIdentity_71:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_72IdentityRestoreV2:tensors:72"/device:CPU:0*
T0*
_output_shapes
:Ц
AssignVariableOp_72AssignVariableOp%assignvariableop_72_conv2d_7_kernel_mIdentity_72:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_73IdentityRestoreV2:tensors:73"/device:CPU:0*
T0*
_output_shapes
:Ф
AssignVariableOp_73AssignVariableOp#assignvariableop_73_conv2d_7_bias_mIdentity_73:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_74IdentityRestoreV2:tensors:74"/device:CPU:0*
T0*
_output_shapes
:Ф
AssignVariableOp_74AssignVariableOp#assignvariableop_74_conv2d_kernel_vIdentity_74:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_75IdentityRestoreV2:tensors:75"/device:CPU:0*
T0*
_output_shapes
:Т
AssignVariableOp_75AssignVariableOp!assignvariableop_75_conv2d_bias_vIdentity_75:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_76IdentityRestoreV2:tensors:76"/device:CPU:0*
T0*
_output_shapes
:†
AssignVariableOp_76AssignVariableOp/assignvariableop_76_batch_normalization_gamma_vIdentity_76:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_77IdentityRestoreV2:tensors:77"/device:CPU:0*
T0*
_output_shapes
:Я
AssignVariableOp_77AssignVariableOp.assignvariableop_77_batch_normalization_beta_vIdentity_77:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_78IdentityRestoreV2:tensors:78"/device:CPU:0*
T0*
_output_shapes
:Ц
AssignVariableOp_78AssignVariableOp%assignvariableop_78_conv2d_1_kernel_vIdentity_78:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_79IdentityRestoreV2:tensors:79"/device:CPU:0*
T0*
_output_shapes
:Ф
AssignVariableOp_79AssignVariableOp#assignvariableop_79_conv2d_1_bias_vIdentity_79:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_80IdentityRestoreV2:tensors:80"/device:CPU:0*
T0*
_output_shapes
:Ґ
AssignVariableOp_80AssignVariableOp1assignvariableop_80_batch_normalization_1_gamma_vIdentity_80:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_81IdentityRestoreV2:tensors:81"/device:CPU:0*
T0*
_output_shapes
:°
AssignVariableOp_81AssignVariableOp0assignvariableop_81_batch_normalization_1_beta_vIdentity_81:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_82IdentityRestoreV2:tensors:82"/device:CPU:0*
T0*
_output_shapes
:Ц
AssignVariableOp_82AssignVariableOp%assignvariableop_82_conv2d_2_kernel_vIdentity_82:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_83IdentityRestoreV2:tensors:83"/device:CPU:0*
T0*
_output_shapes
:Ф
AssignVariableOp_83AssignVariableOp#assignvariableop_83_conv2d_2_bias_vIdentity_83:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_84IdentityRestoreV2:tensors:84"/device:CPU:0*
T0*
_output_shapes
:Ґ
AssignVariableOp_84AssignVariableOp1assignvariableop_84_batch_normalization_2_gamma_vIdentity_84:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_85IdentityRestoreV2:tensors:85"/device:CPU:0*
T0*
_output_shapes
:°
AssignVariableOp_85AssignVariableOp0assignvariableop_85_batch_normalization_2_beta_vIdentity_85:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_86IdentityRestoreV2:tensors:86"/device:CPU:0*
T0*
_output_shapes
:Ц
AssignVariableOp_86AssignVariableOp%assignvariableop_86_conv2d_3_kernel_vIdentity_86:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_87IdentityRestoreV2:tensors:87"/device:CPU:0*
T0*
_output_shapes
:Ф
AssignVariableOp_87AssignVariableOp#assignvariableop_87_conv2d_3_bias_vIdentity_87:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_88IdentityRestoreV2:tensors:88"/device:CPU:0*
T0*
_output_shapes
:Ґ
AssignVariableOp_88AssignVariableOp1assignvariableop_88_batch_normalization_3_gamma_vIdentity_88:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_89IdentityRestoreV2:tensors:89"/device:CPU:0*
T0*
_output_shapes
:°
AssignVariableOp_89AssignVariableOp0assignvariableop_89_batch_normalization_3_beta_vIdentity_89:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_90IdentityRestoreV2:tensors:90"/device:CPU:0*
T0*
_output_shapes
:Ц
AssignVariableOp_90AssignVariableOp%assignvariableop_90_conv2d_4_kernel_vIdentity_90:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_91IdentityRestoreV2:tensors:91"/device:CPU:0*
T0*
_output_shapes
:Ф
AssignVariableOp_91AssignVariableOp#assignvariableop_91_conv2d_4_bias_vIdentity_91:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_92IdentityRestoreV2:tensors:92"/device:CPU:0*
T0*
_output_shapes
:Ґ
AssignVariableOp_92AssignVariableOp1assignvariableop_92_batch_normalization_4_gamma_vIdentity_92:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_93IdentityRestoreV2:tensors:93"/device:CPU:0*
T0*
_output_shapes
:°
AssignVariableOp_93AssignVariableOp0assignvariableop_93_batch_normalization_4_beta_vIdentity_93:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_94IdentityRestoreV2:tensors:94"/device:CPU:0*
T0*
_output_shapes
:Ц
AssignVariableOp_94AssignVariableOp%assignvariableop_94_conv2d_5_kernel_vIdentity_94:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_95IdentityRestoreV2:tensors:95"/device:CPU:0*
T0*
_output_shapes
:Ф
AssignVariableOp_95AssignVariableOp#assignvariableop_95_conv2d_5_bias_vIdentity_95:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_96IdentityRestoreV2:tensors:96"/device:CPU:0*
T0*
_output_shapes
:Ґ
AssignVariableOp_96AssignVariableOp1assignvariableop_96_batch_normalization_5_gamma_vIdentity_96:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_97IdentityRestoreV2:tensors:97"/device:CPU:0*
T0*
_output_shapes
:°
AssignVariableOp_97AssignVariableOp0assignvariableop_97_batch_normalization_5_beta_vIdentity_97:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_98IdentityRestoreV2:tensors:98"/device:CPU:0*
T0*
_output_shapes
:Ц
AssignVariableOp_98AssignVariableOp%assignvariableop_98_conv2d_6_kernel_vIdentity_98:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_99IdentityRestoreV2:tensors:99"/device:CPU:0*
T0*
_output_shapes
:Ф
AssignVariableOp_99AssignVariableOp#assignvariableop_99_conv2d_6_bias_vIdentity_99:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_100IdentityRestoreV2:tensors:100"/device:CPU:0*
T0*
_output_shapes
:•
AssignVariableOp_100AssignVariableOp2assignvariableop_100_batch_normalization_6_gamma_vIdentity_100:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_101IdentityRestoreV2:tensors:101"/device:CPU:0*
T0*
_output_shapes
:§
AssignVariableOp_101AssignVariableOp1assignvariableop_101_batch_normalization_6_beta_vIdentity_101:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_102IdentityRestoreV2:tensors:102"/device:CPU:0*
T0*
_output_shapes
:Щ
AssignVariableOp_102AssignVariableOp&assignvariableop_102_conv2d_7_kernel_vIdentity_102:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_103IdentityRestoreV2:tensors:103"/device:CPU:0*
T0*
_output_shapes
:Ч
AssignVariableOp_103AssignVariableOp$assignvariableop_103_conv2d_7_bias_vIdentity_103:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 ƒ
Identity_104Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_100^AssignVariableOp_101^AssignVariableOp_102^AssignVariableOp_103^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90^AssignVariableOp_91^AssignVariableOp_92^AssignVariableOp_93^AssignVariableOp_94^AssignVariableOp_95^AssignVariableOp_96^AssignVariableOp_97^AssignVariableOp_98^AssignVariableOp_99^NoOp"/device:CPU:0*
T0*
_output_shapes
: Y
Identity_105IdentityIdentity_104:output:0^NoOp_1*
T0*
_output_shapes
: ∞
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_100^AssignVariableOp_101^AssignVariableOp_102^AssignVariableOp_103^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90^AssignVariableOp_91^AssignVariableOp_92^AssignVariableOp_93^AssignVariableOp_94^AssignVariableOp_95^AssignVariableOp_96^AssignVariableOp_97^AssignVariableOp_98^AssignVariableOp_99*"
_acd_function_control_output(*
_output_shapes
 "%
identity_105Identity_105:output:0*з
_input_shapes’
“: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102,
AssignVariableOp_100AssignVariableOp_1002,
AssignVariableOp_101AssignVariableOp_1012,
AssignVariableOp_102AssignVariableOp_1022,
AssignVariableOp_103AssignVariableOp_1032*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_59AssignVariableOp_592(
AssignVariableOp_6AssignVariableOp_62*
AssignVariableOp_60AssignVariableOp_602*
AssignVariableOp_61AssignVariableOp_612*
AssignVariableOp_62AssignVariableOp_622*
AssignVariableOp_63AssignVariableOp_632*
AssignVariableOp_64AssignVariableOp_642*
AssignVariableOp_65AssignVariableOp_652*
AssignVariableOp_66AssignVariableOp_662*
AssignVariableOp_67AssignVariableOp_672*
AssignVariableOp_68AssignVariableOp_682*
AssignVariableOp_69AssignVariableOp_692(
AssignVariableOp_7AssignVariableOp_72*
AssignVariableOp_70AssignVariableOp_702*
AssignVariableOp_71AssignVariableOp_712*
AssignVariableOp_72AssignVariableOp_722*
AssignVariableOp_73AssignVariableOp_732*
AssignVariableOp_74AssignVariableOp_742*
AssignVariableOp_75AssignVariableOp_752*
AssignVariableOp_76AssignVariableOp_762*
AssignVariableOp_77AssignVariableOp_772*
AssignVariableOp_78AssignVariableOp_782*
AssignVariableOp_79AssignVariableOp_792(
AssignVariableOp_8AssignVariableOp_82*
AssignVariableOp_80AssignVariableOp_802*
AssignVariableOp_81AssignVariableOp_812*
AssignVariableOp_82AssignVariableOp_822*
AssignVariableOp_83AssignVariableOp_832*
AssignVariableOp_84AssignVariableOp_842*
AssignVariableOp_85AssignVariableOp_852*
AssignVariableOp_86AssignVariableOp_862*
AssignVariableOp_87AssignVariableOp_872*
AssignVariableOp_88AssignVariableOp_882*
AssignVariableOp_89AssignVariableOp_892(
AssignVariableOp_9AssignVariableOp_92*
AssignVariableOp_90AssignVariableOp_902*
AssignVariableOp_91AssignVariableOp_912*
AssignVariableOp_92AssignVariableOp_922*
AssignVariableOp_93AssignVariableOp_932*
AssignVariableOp_94AssignVariableOp_942*
AssignVariableOp_95AssignVariableOp_952*
AssignVariableOp_96AssignVariableOp_962*
AssignVariableOp_97AssignVariableOp_972*
AssignVariableOp_98AssignVariableOp_982*
AssignVariableOp_99AssignVariableOp_99:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
Ї
B
&__inference_re_lu_6_layer_call_fn_5191

inputs
identityЈ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€@@0* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_re_lu_6_layer_call_and_return_conditional_losses_2073h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:€€€€€€€€€@@0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:€€€€€€€€€@@0:W S
/
_output_shapes
:€€€€€€€€€@@0
 
_user_specified_nameinputs
§
e
I__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_1855

inputs
identityЖ
MaxPoolMaxPoolinputs*/
_output_shapes
:€€€€€€€€€@*
ksize
*
paddingSAME*
strides
`
IdentityIdentityMaxPool:output:0*
T0*/
_output_shapes
:€€€€€€€€€@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:€€€€€€€€€  @:W S
/
_output_shapes
:€€€€€€€€€  @
 
_user_specified_nameinputs
й
]
A__inference_re_lu_4_layer_call_and_return_conditional_losses_4890

inputs
identityO
ReluReluinputs*
T0*0
_output_shapes
:€€€€€€€€€јc
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:€€€€€€€€€ј"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:€€€€€€€€€ј:X T
0
_output_shapes
:€€€€€€€€€ј
 
_user_specified_nameinputs
©

ь
B__inference_conv2d_5_layer_call_and_return_conditional_losses_4909

inputs9
conv2d_readvariableop_resource:а`-
biasadd_readvariableop_resource:`
identityИҐBiasAdd/ReadVariableOpҐConv2D/ReadVariableOp}
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:а`*
dtype0Щ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€  `*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:`*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€  `g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:€€€€€€€€€  `w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :€€€€€€€€€  а: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:€€€€€€€€€  а
 
_user_specified_nameinputs
Ў
Њ
O__inference_batch_normalization_6_layer_call_and_return_conditional_losses_5150

inputs%
readvariableop_resource:0'
readvariableop_1_resource:06
(fusedbatchnormv3_readvariableop_resource:08
*fusedbatchnormv3_readvariableop_1_resource:0
identityИҐAssignNewValueҐAssignNewValue_1ҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:0*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:0*
dtype0Д
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:0*
dtype0И
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:0*
dtype0÷
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+€€€€€€€€€€€€€€€€€€€€€€€€€€€0:0:0:0:0:*
epsilon%oГ:*
exponential_avg_factor%
„#<∞
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0Ї
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€0‘
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+€€€€€€€€€€€€€€€€€€€€€€€€€€€0: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+€€€€€€€€€€€€€€€€€€€€€€€€€€€0
 
_user_specified_nameinputs
Њ
@
$__inference_re_lu_layer_call_fn_4193

inputs
identityЈ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:€€€€€€€€€АА* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *H
fCRA
?__inference_re_lu_layer_call_and_return_conditional_losses_1737j
IdentityIdentityPartitionedCall:output:0*
T0*1
_output_shapes
:€€€€€€€€€АА"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:€€€€€€€€€АА:Y U
1
_output_shapes
:€€€€€€€€€АА
 
_user_specified_nameinputs
Џ
Ю
O__inference_batch_normalization_4_layer_call_and_return_conditional_losses_4826

inputs&
readvariableop_resource:	ј(
readvariableop_1_resource:	ј7
(fusedbatchnormv3_readvariableop_resource:	ј9
*fusedbatchnormv3_readvariableop_1_resource:	ј
identityИҐFusedBatchNormV3/ReadVariableOpҐ!FusedBatchNormV3/ReadVariableOp_1ҐReadVariableOpҐReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:ј*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:ј*
dtype0Е
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:ј*
dtype0Й
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:ј*
dtype0Ќ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,€€€€€€€€€€€€€€€€€€€€€€€€€€€ј:ј:ј:ј:ј:*
epsilon%oГ:*
is_training( ~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€ј∞
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,€€€€€€€€€€€€€€€€€€€€€€€€€€€ј: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,€€€€€€€€€€€€€€€€€€€€€€€€€€€ј
 
_user_specified_nameinputs"ВL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*Њ
serving_default™
E
input_1:
serving_default_input_1:0€€€€€€€€€ААE
reshape:
StatefulPartitionedCall:0€€€€€€€€€ААtensorflow/serving/predict:§Ю
—
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer-4
layer_with_weights-2
layer-5
layer_with_weights-3
layer-6
layer-7
	layer-8

layer_with_weights-4

layer-9
layer_with_weights-5
layer-10
layer-11
layer-12
layer_with_weights-6
layer-13
layer_with_weights-7
layer-14
layer-15
layer-16
layer-17
layer-18
layer_with_weights-8
layer-19
layer_with_weights-9
layer-20
layer-21
layer-22
layer-23
layer_with_weights-10
layer-24
layer_with_weights-11
layer-25
layer-26
layer-27
layer-28
layer_with_weights-12
layer-29
layer_with_weights-13
layer-30
 layer-31
!layer-32
"layer_with_weights-14
"layer-33
#layer-34
$	optimizer
%	variables
&trainable_variables
'regularization_losses
(	keras_api
)
signatures
Ш__call__
+Щ&call_and_return_all_conditional_losses
Ъ_default_save_signature"
_tf_keras_network
"
_tf_keras_input_layer
љ

*kernel
+bias
,	variables
-trainable_variables
.regularization_losses
/	keras_api
Ы__call__
+Ь&call_and_return_all_conditional_losses"
_tf_keras_layer
м
0axis
	1gamma
2beta
3moving_mean
4moving_variance
5	variables
6trainable_variables
7regularization_losses
8	keras_api
Э__call__
+Ю&call_and_return_all_conditional_losses"
_tf_keras_layer
І
9	variables
:trainable_variables
;regularization_losses
<	keras_api
Я__call__
+†&call_and_return_all_conditional_losses"
_tf_keras_layer
І
=	variables
>trainable_variables
?regularization_losses
@	keras_api
°__call__
+Ґ&call_and_return_all_conditional_losses"
_tf_keras_layer
љ

Akernel
Bbias
C	variables
Dtrainable_variables
Eregularization_losses
F	keras_api
£__call__
+§&call_and_return_all_conditional_losses"
_tf_keras_layer
м
Gaxis
	Hgamma
Ibeta
Jmoving_mean
Kmoving_variance
L	variables
Mtrainable_variables
Nregularization_losses
O	keras_api
•__call__
+¶&call_and_return_all_conditional_losses"
_tf_keras_layer
І
P	variables
Qtrainable_variables
Rregularization_losses
S	keras_api
І__call__
+®&call_and_return_all_conditional_losses"
_tf_keras_layer
І
T	variables
Utrainable_variables
Vregularization_losses
W	keras_api
©__call__
+™&call_and_return_all_conditional_losses"
_tf_keras_layer
љ

Xkernel
Ybias
Z	variables
[trainable_variables
\regularization_losses
]	keras_api
Ђ__call__
+ђ&call_and_return_all_conditional_losses"
_tf_keras_layer
м
^axis
	_gamma
`beta
amoving_mean
bmoving_variance
c	variables
dtrainable_variables
eregularization_losses
f	keras_api
≠__call__
+Ѓ&call_and_return_all_conditional_losses"
_tf_keras_layer
І
g	variables
htrainable_variables
iregularization_losses
j	keras_api
ѓ__call__
+∞&call_and_return_all_conditional_losses"
_tf_keras_layer
І
k	variables
ltrainable_variables
mregularization_losses
n	keras_api
±__call__
+≤&call_and_return_all_conditional_losses"
_tf_keras_layer
љ

okernel
pbias
q	variables
rtrainable_variables
sregularization_losses
t	keras_api
≥__call__
+і&call_and_return_all_conditional_losses"
_tf_keras_layer
м
uaxis
	vgamma
wbeta
xmoving_mean
ymoving_variance
z	variables
{trainable_variables
|regularization_losses
}	keras_api
µ__call__
+ґ&call_and_return_all_conditional_losses"
_tf_keras_layer
©
~	variables
trainable_variables
Аregularization_losses
Б	keras_api
Ј__call__
+Є&call_and_return_all_conditional_losses"
_tf_keras_layer
Ђ
В	variables
Гtrainable_variables
Дregularization_losses
Е	keras_api
є__call__
+Ї&call_and_return_all_conditional_losses"
_tf_keras_layer
)
Ж	keras_api"
_tf_keras_layer
)
З	keras_api"
_tf_keras_layer
√
Иkernel
	Йbias
К	variables
Лtrainable_variables
Мregularization_losses
Н	keras_api
ї__call__
+Љ&call_and_return_all_conditional_losses"
_tf_keras_layer
х
	Оaxis

Пgamma
	Рbeta
Сmoving_mean
Тmoving_variance
У	variables
Фtrainable_variables
Хregularization_losses
Ц	keras_api
љ__call__
+Њ&call_and_return_all_conditional_losses"
_tf_keras_layer
Ђ
Ч	variables
Шtrainable_variables
Щregularization_losses
Ъ	keras_api
њ__call__
+ј&call_and_return_all_conditional_losses"
_tf_keras_layer
)
Ы	keras_api"
_tf_keras_layer
)
Ь	keras_api"
_tf_keras_layer
√
Эkernel
	Юbias
Я	variables
†trainable_variables
°regularization_losses
Ґ	keras_api
Ѕ__call__
+¬&call_and_return_all_conditional_losses"
_tf_keras_layer
х
	£axis

§gamma
	•beta
¶moving_mean
Іmoving_variance
®	variables
©trainable_variables
™regularization_losses
Ђ	keras_api
√__call__
+ƒ&call_and_return_all_conditional_losses"
_tf_keras_layer
Ђ
ђ	variables
≠trainable_variables
Ѓregularization_losses
ѓ	keras_api
≈__call__
+∆&call_and_return_all_conditional_losses"
_tf_keras_layer
)
∞	keras_api"
_tf_keras_layer
)
±	keras_api"
_tf_keras_layer
√
≤kernel
	≥bias
і	variables
µtrainable_variables
ґregularization_losses
Ј	keras_api
«__call__
+»&call_and_return_all_conditional_losses"
_tf_keras_layer
х
	Єaxis

єgamma
	Їbeta
їmoving_mean
Љmoving_variance
љ	variables
Њtrainable_variables
њregularization_losses
ј	keras_api
…__call__
+ &call_and_return_all_conditional_losses"
_tf_keras_layer
Ђ
Ѕ	variables
¬trainable_variables
√regularization_losses
ƒ	keras_api
Ћ__call__
+ћ&call_and_return_all_conditional_losses"
_tf_keras_layer
)
≈	keras_api"
_tf_keras_layer
√
∆kernel
	«bias
»	variables
…trainable_variables
 regularization_losses
Ћ	keras_api
Ќ__call__
+ќ&call_and_return_all_conditional_losses"
_tf_keras_layer
Ђ
ћ	variables
Ќtrainable_variables
ќregularization_losses
ѕ	keras_api
ѕ__call__
+–&call_and_return_all_conditional_losses"
_tf_keras_layer
З*m№+mЁ1mё2mяAmаBmбHmвImгXmдYmе_mж`mзomиpmйvmкwmл	Иmм	Йmн	Пmо	Рmп	Эmр	Юmс	§mт	•mу	≤mф	≥mх	єmц	Їmч	∆mш	«mщ*vъ+vы1vь2vэAvюBv€HvАIvБXvВYvГ_vД`vЕovЖpvЗvvИwvЙ	ИvК	ЙvЛ	ПvМ	РvН	ЭvО	ЮvП	§vР	•vС	≤vТ	≥vУ	єvФ	ЇvХ	∆vЦ	«vЧ"
	optimizer
К
*0
+1
12
23
34
45
A6
B7
H8
I9
J10
K11
X12
Y13
_14
`15
a16
b17
o18
p19
v20
w21
x22
y23
И24
Й25
П26
Р27
С28
Т29
Э30
Ю31
§32
•33
¶34
І35
≤36
≥37
є38
Ї39
ї40
Љ41
∆42
«43"
trackable_list_wrapper
Ф
*0
+1
12
23
A4
B5
H6
I7
X8
Y9
_10
`11
o12
p13
v14
w15
И16
Й17
П18
Р19
Э20
Ю21
§22
•23
≤24
≥25
є26
Ї27
∆28
«29"
trackable_list_wrapper
 "
trackable_list_wrapper
”
–non_trainable_variables
—layers
“metrics
 ”layer_regularization_losses
‘layer_metrics
%	variables
&trainable_variables
'regularization_losses
Ш__call__
Ъ_default_save_signature
+Щ&call_and_return_all_conditional_losses
'Щ"call_and_return_conditional_losses"
_generic_user_object
-
—serving_default"
signature_map
':%2conv2d/kernel
:2conv2d/bias
.
*0
+1"
trackable_list_wrapper
.
*0
+1"
trackable_list_wrapper
 "
trackable_list_wrapper
µ
’non_trainable_variables
÷layers
„metrics
 Ўlayer_regularization_losses
ўlayer_metrics
,	variables
-trainable_variables
.regularization_losses
Ы__call__
+Ь&call_and_return_all_conditional_losses
'Ь"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
':%2batch_normalization/gamma
&:$2batch_normalization/beta
/:- (2batch_normalization/moving_mean
3:1 (2#batch_normalization/moving_variance
<
10
21
32
43"
trackable_list_wrapper
.
10
21"
trackable_list_wrapper
 "
trackable_list_wrapper
µ
Џnon_trainable_variables
џlayers
№metrics
 Ёlayer_regularization_losses
ёlayer_metrics
5	variables
6trainable_variables
7regularization_losses
Э__call__
+Ю&call_and_return_all_conditional_losses
'Ю"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
яnon_trainable_variables
аlayers
бmetrics
 вlayer_regularization_losses
гlayer_metrics
9	variables
:trainable_variables
;regularization_losses
Я__call__
+†&call_and_return_all_conditional_losses
'†"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
дnon_trainable_variables
еlayers
жmetrics
 зlayer_regularization_losses
иlayer_metrics
=	variables
>trainable_variables
?regularization_losses
°__call__
+Ґ&call_and_return_all_conditional_losses
'Ґ"call_and_return_conditional_losses"
_generic_user_object
):' 2conv2d_1/kernel
: 2conv2d_1/bias
.
A0
B1"
trackable_list_wrapper
.
A0
B1"
trackable_list_wrapper
 "
trackable_list_wrapper
µ
йnon_trainable_variables
кlayers
лmetrics
 мlayer_regularization_losses
нlayer_metrics
C	variables
Dtrainable_variables
Eregularization_losses
£__call__
+§&call_and_return_all_conditional_losses
'§"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
):' 2batch_normalization_1/gamma
(:& 2batch_normalization_1/beta
1:/  (2!batch_normalization_1/moving_mean
5:3  (2%batch_normalization_1/moving_variance
<
H0
I1
J2
K3"
trackable_list_wrapper
.
H0
I1"
trackable_list_wrapper
 "
trackable_list_wrapper
µ
оnon_trainable_variables
пlayers
рmetrics
 сlayer_regularization_losses
тlayer_metrics
L	variables
Mtrainable_variables
Nregularization_losses
•__call__
+¶&call_and_return_all_conditional_losses
'¶"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
уnon_trainable_variables
фlayers
хmetrics
 цlayer_regularization_losses
чlayer_metrics
P	variables
Qtrainable_variables
Rregularization_losses
І__call__
+®&call_and_return_all_conditional_losses
'®"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
шnon_trainable_variables
щlayers
ъmetrics
 ыlayer_regularization_losses
ьlayer_metrics
T	variables
Utrainable_variables
Vregularization_losses
©__call__
+™&call_and_return_all_conditional_losses
'™"call_and_return_conditional_losses"
_generic_user_object
):' @2conv2d_2/kernel
:@2conv2d_2/bias
.
X0
Y1"
trackable_list_wrapper
.
X0
Y1"
trackable_list_wrapper
 "
trackable_list_wrapper
µ
эnon_trainable_variables
юlayers
€metrics
 Аlayer_regularization_losses
Бlayer_metrics
Z	variables
[trainable_variables
\regularization_losses
Ђ__call__
+ђ&call_and_return_all_conditional_losses
'ђ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
):'@2batch_normalization_2/gamma
(:&@2batch_normalization_2/beta
1:/@ (2!batch_normalization_2/moving_mean
5:3@ (2%batch_normalization_2/moving_variance
<
_0
`1
a2
b3"
trackable_list_wrapper
.
_0
`1"
trackable_list_wrapper
 "
trackable_list_wrapper
µ
Вnon_trainable_variables
Гlayers
Дmetrics
 Еlayer_regularization_losses
Жlayer_metrics
c	variables
dtrainable_variables
eregularization_losses
≠__call__
+Ѓ&call_and_return_all_conditional_losses
'Ѓ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
Зnon_trainable_variables
Иlayers
Йmetrics
 Кlayer_regularization_losses
Лlayer_metrics
g	variables
htrainable_variables
iregularization_losses
ѓ__call__
+∞&call_and_return_all_conditional_losses
'∞"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
Мnon_trainable_variables
Нlayers
Оmetrics
 Пlayer_regularization_losses
Рlayer_metrics
k	variables
ltrainable_variables
mregularization_losses
±__call__
+≤&call_and_return_all_conditional_losses
'≤"call_and_return_conditional_losses"
_generic_user_object
*:(@А2conv2d_3/kernel
:А2conv2d_3/bias
.
o0
p1"
trackable_list_wrapper
.
o0
p1"
trackable_list_wrapper
 "
trackable_list_wrapper
µ
Сnon_trainable_variables
Тlayers
Уmetrics
 Фlayer_regularization_losses
Хlayer_metrics
q	variables
rtrainable_variables
sregularization_losses
≥__call__
+і&call_and_return_all_conditional_losses
'і"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
*:(А2batch_normalization_3/gamma
):'А2batch_normalization_3/beta
2:0А (2!batch_normalization_3/moving_mean
6:4А (2%batch_normalization_3/moving_variance
<
v0
w1
x2
y3"
trackable_list_wrapper
.
v0
w1"
trackable_list_wrapper
 "
trackable_list_wrapper
µ
Цnon_trainable_variables
Чlayers
Шmetrics
 Щlayer_regularization_losses
Ъlayer_metrics
z	variables
{trainable_variables
|regularization_losses
µ__call__
+ґ&call_and_return_all_conditional_losses
'ґ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
ґ
Ыnon_trainable_variables
Ьlayers
Эmetrics
 Юlayer_regularization_losses
Яlayer_metrics
~	variables
trainable_variables
Аregularization_losses
Ј__call__
+Є&call_and_return_all_conditional_losses
'Є"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Є
†non_trainable_variables
°layers
Ґmetrics
 £layer_regularization_losses
§layer_metrics
В	variables
Гtrainable_variables
Дregularization_losses
є__call__
+Ї&call_and_return_all_conditional_losses
'Ї"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
+:)јј2conv2d_4/kernel
:ј2conv2d_4/bias
0
И0
Й1"
trackable_list_wrapper
0
И0
Й1"
trackable_list_wrapper
 "
trackable_list_wrapper
Є
•non_trainable_variables
¶layers
Іmetrics
 ®layer_regularization_losses
©layer_metrics
К	variables
Лtrainable_variables
Мregularization_losses
ї__call__
+Љ&call_and_return_all_conditional_losses
'Љ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
*:(ј2batch_normalization_4/gamma
):'ј2batch_normalization_4/beta
2:0ј (2!batch_normalization_4/moving_mean
6:4ј (2%batch_normalization_4/moving_variance
@
П0
Р1
С2
Т3"
trackable_list_wrapper
0
П0
Р1"
trackable_list_wrapper
 "
trackable_list_wrapper
Є
™non_trainable_variables
Ђlayers
ђmetrics
 ≠layer_regularization_losses
Ѓlayer_metrics
У	variables
Фtrainable_variables
Хregularization_losses
љ__call__
+Њ&call_and_return_all_conditional_losses
'Њ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Є
ѓnon_trainable_variables
∞layers
±metrics
 ≤layer_regularization_losses
≥layer_metrics
Ч	variables
Шtrainable_variables
Щregularization_losses
њ__call__
+ј&call_and_return_all_conditional_losses
'ј"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
*:(а`2conv2d_5/kernel
:`2conv2d_5/bias
0
Э0
Ю1"
trackable_list_wrapper
0
Э0
Ю1"
trackable_list_wrapper
 "
trackable_list_wrapper
Є
іnon_trainable_variables
µlayers
ґmetrics
 Јlayer_regularization_losses
Єlayer_metrics
Я	variables
†trainable_variables
°regularization_losses
Ѕ__call__
+¬&call_and_return_all_conditional_losses
'¬"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
):'`2batch_normalization_5/gamma
(:&`2batch_normalization_5/beta
1:/` (2!batch_normalization_5/moving_mean
5:3` (2%batch_normalization_5/moving_variance
@
§0
•1
¶2
І3"
trackable_list_wrapper
0
§0
•1"
trackable_list_wrapper
 "
trackable_list_wrapper
Є
єnon_trainable_variables
Їlayers
їmetrics
 Љlayer_regularization_losses
љlayer_metrics
®	variables
©trainable_variables
™regularization_losses
√__call__
+ƒ&call_and_return_all_conditional_losses
'ƒ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Є
Њnon_trainable_variables
њlayers
јmetrics
 Ѕlayer_regularization_losses
¬layer_metrics
ђ	variables
≠trainable_variables
Ѓregularization_losses
≈__call__
+∆&call_and_return_all_conditional_losses
'∆"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
):'p02conv2d_6/kernel
:02conv2d_6/bias
0
≤0
≥1"
trackable_list_wrapper
0
≤0
≥1"
trackable_list_wrapper
 "
trackable_list_wrapper
Є
√non_trainable_variables
ƒlayers
≈metrics
 ∆layer_regularization_losses
«layer_metrics
і	variables
µtrainable_variables
ґregularization_losses
«__call__
+»&call_and_return_all_conditional_losses
'»"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
):'02batch_normalization_6/gamma
(:&02batch_normalization_6/beta
1:/0 (2!batch_normalization_6/moving_mean
5:30 (2%batch_normalization_6/moving_variance
@
є0
Ї1
ї2
Љ3"
trackable_list_wrapper
0
є0
Ї1"
trackable_list_wrapper
 "
trackable_list_wrapper
Є
»non_trainable_variables
…layers
 metrics
 Ћlayer_regularization_losses
ћlayer_metrics
љ	variables
Њtrainable_variables
њregularization_losses
…__call__
+ &call_and_return_all_conditional_losses
' "call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Є
Ќnon_trainable_variables
ќlayers
ѕmetrics
 –layer_regularization_losses
—layer_metrics
Ѕ	variables
¬trainable_variables
√regularization_losses
Ћ__call__
+ћ&call_and_return_all_conditional_losses
'ћ"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
):'02conv2d_7/kernel
:2conv2d_7/bias
0
∆0
«1"
trackable_list_wrapper
0
∆0
«1"
trackable_list_wrapper
 "
trackable_list_wrapper
Є
“non_trainable_variables
”layers
‘metrics
 ’layer_regularization_losses
÷layer_metrics
»	variables
…trainable_variables
 regularization_losses
Ќ__call__
+ќ&call_and_return_all_conditional_losses
'ќ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Є
„non_trainable_variables
Ўlayers
ўmetrics
 Џlayer_regularization_losses
џlayer_metrics
ћ	variables
Ќtrainable_variables
ќregularization_losses
ѕ__call__
+–&call_and_return_all_conditional_losses
'–"call_and_return_conditional_losses"
_generic_user_object
М
30
41
J2
K3
a4
b5
x6
y7
С8
Т9
¶10
І11
ї12
Љ13"
trackable_list_wrapper
Ѓ
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
30
 31
!32
"33
#34"
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
30
41"
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
J0
K1"
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
a0
b1"
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
x0
y1"
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
0
С0
Т1"
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
0
¶0
І1"
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
0
ї0
Љ1"
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
':%2conv2d/kernel/m
:2conv2d/bias/m
':%2batch_normalization/gamma/m
&:$2batch_normalization/beta/m
):' 2conv2d_1/kernel/m
: 2conv2d_1/bias/m
):' 2batch_normalization_1/gamma/m
(:& 2batch_normalization_1/beta/m
):' @2conv2d_2/kernel/m
:@2conv2d_2/bias/m
):'@2batch_normalization_2/gamma/m
(:&@2batch_normalization_2/beta/m
*:(@А2conv2d_3/kernel/m
:А2conv2d_3/bias/m
*:(А2batch_normalization_3/gamma/m
):'А2batch_normalization_3/beta/m
+:)јј2conv2d_4/kernel/m
:ј2conv2d_4/bias/m
*:(ј2batch_normalization_4/gamma/m
):'ј2batch_normalization_4/beta/m
*:(а`2conv2d_5/kernel/m
:`2conv2d_5/bias/m
):'`2batch_normalization_5/gamma/m
(:&`2batch_normalization_5/beta/m
):'p02conv2d_6/kernel/m
:02conv2d_6/bias/m
):'02batch_normalization_6/gamma/m
(:&02batch_normalization_6/beta/m
):'02conv2d_7/kernel/m
:2conv2d_7/bias/m
':%2conv2d/kernel/v
:2conv2d/bias/v
':%2batch_normalization/gamma/v
&:$2batch_normalization/beta/v
):' 2conv2d_1/kernel/v
: 2conv2d_1/bias/v
):' 2batch_normalization_1/gamma/v
(:& 2batch_normalization_1/beta/v
):' @2conv2d_2/kernel/v
:@2conv2d_2/bias/v
):'@2batch_normalization_2/gamma/v
(:&@2batch_normalization_2/beta/v
*:(@А2conv2d_3/kernel/v
:А2conv2d_3/bias/v
*:(А2batch_normalization_3/gamma/v
):'А2batch_normalization_3/beta/v
+:)јј2conv2d_4/kernel/v
:ј2conv2d_4/bias/v
*:(ј2batch_normalization_4/gamma/v
):'ј2batch_normalization_4/beta/v
*:(а`2conv2d_5/kernel/v
:`2conv2d_5/bias/v
):'`2batch_normalization_5/gamma/v
(:&`2batch_normalization_5/beta/v
):'p02conv2d_6/kernel/v
:02conv2d_6/bias/v
):'02batch_normalization_6/gamma/v
(:&02batch_normalization_6/beta/v
):'02conv2d_7/kernel/v
:2conv2d_7/bias/v
ц2у
*__inference_Mobile_Unet_layer_call_fn_2210
*__inference_Mobile_Unet_layer_call_fn_3572
*__inference_Mobile_Unet_layer_call_fn_3673
*__inference_Mobile_Unet_layer_call_fn_3102ј
Ј≤≥
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
в2я
E__inference_Mobile_Unet_layer_call_and_return_conditional_losses_3859
E__inference_Mobile_Unet_layer_call_and_return_conditional_losses_4045
E__inference_Mobile_Unet_layer_call_and_return_conditional_losses_3235
E__inference_Mobile_Unet_layer_call_and_return_conditional_losses_3368ј
Ј≤≥
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
 B«
__inference__wrapped_model_1186input_1"Ш
С≤Н
FullArgSpec
argsЪ 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ѕ2ћ
%__inference_conv2d_layer_call_fn_4054Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
к2з
@__inference_conv2d_layer_call_and_return_conditional_losses_4064Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
К2З
2__inference_batch_normalization_layer_call_fn_4077
2__inference_batch_normalization_layer_call_fn_4090
2__inference_batch_normalization_layer_call_fn_4103
2__inference_batch_normalization_layer_call_fn_4116і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
ц2у
M__inference_batch_normalization_layer_call_and_return_conditional_losses_4134
M__inference_batch_normalization_layer_call_and_return_conditional_losses_4152
M__inference_batch_normalization_layer_call_and_return_conditional_losses_4170
M__inference_batch_normalization_layer_call_and_return_conditional_losses_4188і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
ќ2Ћ
$__inference_re_lu_layer_call_fn_4193Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
й2ж
?__inference_re_lu_layer_call_and_return_conditional_losses_4198Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
Д2Б
,__inference_max_pooling2d_layer_call_fn_4203
,__inference_max_pooling2d_layer_call_fn_4208Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
Ї2Ј
G__inference_max_pooling2d_layer_call_and_return_conditional_losses_4213
G__inference_max_pooling2d_layer_call_and_return_conditional_losses_4218Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
—2ќ
'__inference_conv2d_1_layer_call_fn_4227Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
м2й
B__inference_conv2d_1_layer_call_and_return_conditional_losses_4237Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
Т2П
4__inference_batch_normalization_1_layer_call_fn_4250
4__inference_batch_normalization_1_layer_call_fn_4263
4__inference_batch_normalization_1_layer_call_fn_4276
4__inference_batch_normalization_1_layer_call_fn_4289і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
ю2ы
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_4307
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_4325
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_4343
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_4361і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
–2Ќ
&__inference_re_lu_1_layer_call_fn_4366Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
л2и
A__inference_re_lu_1_layer_call_and_return_conditional_losses_4371Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
И2Е
.__inference_max_pooling2d_1_layer_call_fn_4376
.__inference_max_pooling2d_1_layer_call_fn_4381Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
Њ2ї
I__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_4386
I__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_4391Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
—2ќ
'__inference_conv2d_2_layer_call_fn_4400Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
м2й
B__inference_conv2d_2_layer_call_and_return_conditional_losses_4410Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
Т2П
4__inference_batch_normalization_2_layer_call_fn_4423
4__inference_batch_normalization_2_layer_call_fn_4436
4__inference_batch_normalization_2_layer_call_fn_4449
4__inference_batch_normalization_2_layer_call_fn_4462і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
ю2ы
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_4480
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_4498
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_4516
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_4534і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
–2Ќ
&__inference_re_lu_2_layer_call_fn_4539Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
л2и
A__inference_re_lu_2_layer_call_and_return_conditional_losses_4544Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
И2Е
.__inference_max_pooling2d_2_layer_call_fn_4549
.__inference_max_pooling2d_2_layer_call_fn_4554Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
Њ2ї
I__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_4559
I__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_4564Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
—2ќ
'__inference_conv2d_3_layer_call_fn_4573Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
м2й
B__inference_conv2d_3_layer_call_and_return_conditional_losses_4583Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
Т2П
4__inference_batch_normalization_3_layer_call_fn_4596
4__inference_batch_normalization_3_layer_call_fn_4609
4__inference_batch_normalization_3_layer_call_fn_4622
4__inference_batch_normalization_3_layer_call_fn_4635і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
ю2ы
O__inference_batch_normalization_3_layer_call_and_return_conditional_losses_4653
O__inference_batch_normalization_3_layer_call_and_return_conditional_losses_4671
O__inference_batch_normalization_3_layer_call_and_return_conditional_losses_4689
O__inference_batch_normalization_3_layer_call_and_return_conditional_losses_4707і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
–2Ќ
&__inference_re_lu_3_layer_call_fn_4712Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
л2и
A__inference_re_lu_3_layer_call_and_return_conditional_losses_4717Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
И2Е
.__inference_max_pooling2d_3_layer_call_fn_4722
.__inference_max_pooling2d_3_layer_call_fn_4727Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
Њ2ї
I__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_4732
I__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_4737Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
—2ќ
'__inference_conv2d_4_layer_call_fn_4746Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
м2й
B__inference_conv2d_4_layer_call_and_return_conditional_losses_4756Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
Т2П
4__inference_batch_normalization_4_layer_call_fn_4769
4__inference_batch_normalization_4_layer_call_fn_4782
4__inference_batch_normalization_4_layer_call_fn_4795
4__inference_batch_normalization_4_layer_call_fn_4808і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
ю2ы
O__inference_batch_normalization_4_layer_call_and_return_conditional_losses_4826
O__inference_batch_normalization_4_layer_call_and_return_conditional_losses_4844
O__inference_batch_normalization_4_layer_call_and_return_conditional_losses_4862
O__inference_batch_normalization_4_layer_call_and_return_conditional_losses_4880і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
–2Ќ
&__inference_re_lu_4_layer_call_fn_4885Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
л2и
A__inference_re_lu_4_layer_call_and_return_conditional_losses_4890Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
—2ќ
'__inference_conv2d_5_layer_call_fn_4899Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
м2й
B__inference_conv2d_5_layer_call_and_return_conditional_losses_4909Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
Т2П
4__inference_batch_normalization_5_layer_call_fn_4922
4__inference_batch_normalization_5_layer_call_fn_4935
4__inference_batch_normalization_5_layer_call_fn_4948
4__inference_batch_normalization_5_layer_call_fn_4961і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
ю2ы
O__inference_batch_normalization_5_layer_call_and_return_conditional_losses_4979
O__inference_batch_normalization_5_layer_call_and_return_conditional_losses_4997
O__inference_batch_normalization_5_layer_call_and_return_conditional_losses_5015
O__inference_batch_normalization_5_layer_call_and_return_conditional_losses_5033і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
–2Ќ
&__inference_re_lu_5_layer_call_fn_5038Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
л2и
A__inference_re_lu_5_layer_call_and_return_conditional_losses_5043Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
—2ќ
'__inference_conv2d_6_layer_call_fn_5052Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
м2й
B__inference_conv2d_6_layer_call_and_return_conditional_losses_5062Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
Т2П
4__inference_batch_normalization_6_layer_call_fn_5075
4__inference_batch_normalization_6_layer_call_fn_5088
4__inference_batch_normalization_6_layer_call_fn_5101
4__inference_batch_normalization_6_layer_call_fn_5114і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
ю2ы
O__inference_batch_normalization_6_layer_call_and_return_conditional_losses_5132
O__inference_batch_normalization_6_layer_call_and_return_conditional_losses_5150
O__inference_batch_normalization_6_layer_call_and_return_conditional_losses_5168
O__inference_batch_normalization_6_layer_call_and_return_conditional_losses_5186і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
–2Ќ
&__inference_re_lu_6_layer_call_fn_5191Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
л2и
A__inference_re_lu_6_layer_call_and_return_conditional_losses_5196Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
—2ќ
'__inference_conv2d_7_layer_call_fn_5205Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
м2й
B__inference_conv2d_7_layer_call_and_return_conditional_losses_5216Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
–2Ќ
&__inference_reshape_layer_call_fn_5221Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
л2и
A__inference_reshape_layer_call_and_return_conditional_losses_5235Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
…B∆
"__inference_signature_wrapper_3471input_1"Ф
Н≤Й
FullArgSpec
argsЪ 
varargs
 
varkwjkwargs
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
	J
Const
J	
Const_1
J	
Const_2
J	
Const_3Й
E__inference_Mobile_Unet_layer_call_and_return_conditional_losses_3235њH*+1234ABHIJKXY_`abopvwxy“ИЙПРСТ”ЭЮ§•¶І‘≤≥єЇїЉ’∆«BҐ?
8Ґ5
+К(
input_1€€€€€€€€€АА
p 

 
™ "/Ґ,
%К"
0€€€€€€€€€АА
Ъ Й
E__inference_Mobile_Unet_layer_call_and_return_conditional_losses_3368њH*+1234ABHIJKXY_`abopvwxy“ИЙПРСТ”ЭЮ§•¶І‘≤≥єЇїЉ’∆«BҐ?
8Ґ5
+К(
input_1€€€€€€€€€АА
p

 
™ "/Ґ,
%К"
0€€€€€€€€€АА
Ъ И
E__inference_Mobile_Unet_layer_call_and_return_conditional_losses_3859ЊH*+1234ABHIJKXY_`abopvwxy“ИЙПРСТ”ЭЮ§•¶І‘≤≥єЇїЉ’∆«AҐ>
7Ґ4
*К'
inputs€€€€€€€€€АА
p 

 
™ "/Ґ,
%К"
0€€€€€€€€€АА
Ъ И
E__inference_Mobile_Unet_layer_call_and_return_conditional_losses_4045ЊH*+1234ABHIJKXY_`abopvwxy“ИЙПРСТ”ЭЮ§•¶І‘≤≥єЇїЉ’∆«AҐ>
7Ґ4
*К'
inputs€€€€€€€€€АА
p

 
™ "/Ґ,
%К"
0€€€€€€€€€АА
Ъ б
*__inference_Mobile_Unet_layer_call_fn_2210≤H*+1234ABHIJKXY_`abopvwxy“ИЙПРСТ”ЭЮ§•¶І‘≤≥єЇїЉ’∆«BҐ?
8Ґ5
+К(
input_1€€€€€€€€€АА
p 

 
™ ""К€€€€€€€€€ААб
*__inference_Mobile_Unet_layer_call_fn_3102≤H*+1234ABHIJKXY_`abopvwxy“ИЙПРСТ”ЭЮ§•¶І‘≤≥єЇїЉ’∆«BҐ?
8Ґ5
+К(
input_1€€€€€€€€€АА
p

 
™ ""К€€€€€€€€€ААа
*__inference_Mobile_Unet_layer_call_fn_3572±H*+1234ABHIJKXY_`abopvwxy“ИЙПРСТ”ЭЮ§•¶І‘≤≥єЇїЉ’∆«AҐ>
7Ґ4
*К'
inputs€€€€€€€€€АА
p 

 
™ ""К€€€€€€€€€ААа
*__inference_Mobile_Unet_layer_call_fn_3673±H*+1234ABHIJKXY_`abopvwxy“ИЙПРСТ”ЭЮ§•¶І‘≤≥єЇїЉ’∆«AҐ>
7Ґ4
*К'
inputs€€€€€€€€€АА
p

 
™ ""К€€€€€€€€€ААз
__inference__wrapped_model_1186√H*+1234ABHIJKXY_`abopvwxy“ИЙПРСТ”ЭЮ§•¶І‘≤≥єЇїЉ’∆«:Ґ7
0Ґ-
+К(
input_1€€€€€€€€€АА
™ ";™8
6
reshape+К(
reshape€€€€€€€€€ААк
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_4307ЦHIJKMҐJ
CҐ@
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 
p 
™ "?Ґ<
5К2
0+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 
Ъ к
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_4325ЦHIJKMҐJ
CҐ@
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 
p
™ "?Ґ<
5К2
0+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 
Ъ ≈
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_4343rHIJK;Ґ8
1Ґ.
(К%
inputs€€€€€€€€€@@ 
p 
™ "-Ґ*
#К 
0€€€€€€€€€@@ 
Ъ ≈
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_4361rHIJK;Ґ8
1Ґ.
(К%
inputs€€€€€€€€€@@ 
p
™ "-Ґ*
#К 
0€€€€€€€€€@@ 
Ъ ¬
4__inference_batch_normalization_1_layer_call_fn_4250ЙHIJKMҐJ
CҐ@
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 
p 
™ "2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€ ¬
4__inference_batch_normalization_1_layer_call_fn_4263ЙHIJKMҐJ
CҐ@
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€ 
p
™ "2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€ Э
4__inference_batch_normalization_1_layer_call_fn_4276eHIJK;Ґ8
1Ґ.
(К%
inputs€€€€€€€€€@@ 
p 
™ " К€€€€€€€€€@@ Э
4__inference_batch_normalization_1_layer_call_fn_4289eHIJK;Ґ8
1Ґ.
(К%
inputs€€€€€€€€€@@ 
p
™ " К€€€€€€€€€@@ к
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_4480Ц_`abMҐJ
CҐ@
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
p 
™ "?Ґ<
5К2
0+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
Ъ к
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_4498Ц_`abMҐJ
CҐ@
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
p
™ "?Ґ<
5К2
0+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
Ъ ≈
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_4516r_`ab;Ґ8
1Ґ.
(К%
inputs€€€€€€€€€  @
p 
™ "-Ґ*
#К 
0€€€€€€€€€  @
Ъ ≈
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_4534r_`ab;Ґ8
1Ґ.
(К%
inputs€€€€€€€€€  @
p
™ "-Ґ*
#К 
0€€€€€€€€€  @
Ъ ¬
4__inference_batch_normalization_2_layer_call_fn_4423Й_`abMҐJ
CҐ@
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
p 
™ "2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€@¬
4__inference_batch_normalization_2_layer_call_fn_4436Й_`abMҐJ
CҐ@
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€@
p
™ "2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€@Э
4__inference_batch_normalization_2_layer_call_fn_4449e_`ab;Ґ8
1Ґ.
(К%
inputs€€€€€€€€€  @
p 
™ " К€€€€€€€€€  @Э
4__inference_batch_normalization_2_layer_call_fn_4462e_`ab;Ґ8
1Ґ.
(К%
inputs€€€€€€€€€  @
p
™ " К€€€€€€€€€  @м
O__inference_batch_normalization_3_layer_call_and_return_conditional_losses_4653ШvwxyNҐK
DҐA
;К8
inputs,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
p 
™ "@Ґ=
6К3
0,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
Ъ м
O__inference_batch_normalization_3_layer_call_and_return_conditional_losses_4671ШvwxyNҐK
DҐA
;К8
inputs,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
p
™ "@Ґ=
6К3
0,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
Ъ «
O__inference_batch_normalization_3_layer_call_and_return_conditional_losses_4689tvwxy<Ґ9
2Ґ/
)К&
inputs€€€€€€€€€А
p 
™ ".Ґ+
$К!
0€€€€€€€€€А
Ъ «
O__inference_batch_normalization_3_layer_call_and_return_conditional_losses_4707tvwxy<Ґ9
2Ґ/
)К&
inputs€€€€€€€€€А
p
™ ".Ґ+
$К!
0€€€€€€€€€А
Ъ ƒ
4__inference_batch_normalization_3_layer_call_fn_4596ЛvwxyNҐK
DҐA
;К8
inputs,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
p 
™ "3К0,€€€€€€€€€€€€€€€€€€€€€€€€€€€Аƒ
4__inference_batch_normalization_3_layer_call_fn_4609ЛvwxyNҐK
DҐA
;К8
inputs,€€€€€€€€€€€€€€€€€€€€€€€€€€€А
p
™ "3К0,€€€€€€€€€€€€€€€€€€€€€€€€€€€АЯ
4__inference_batch_normalization_3_layer_call_fn_4622gvwxy<Ґ9
2Ґ/
)К&
inputs€€€€€€€€€А
p 
™ "!К€€€€€€€€€АЯ
4__inference_batch_normalization_3_layer_call_fn_4635gvwxy<Ґ9
2Ґ/
)К&
inputs€€€€€€€€€А
p
™ "!К€€€€€€€€€Ар
O__inference_batch_normalization_4_layer_call_and_return_conditional_losses_4826ЬПРСТNҐK
DҐA
;К8
inputs,€€€€€€€€€€€€€€€€€€€€€€€€€€€ј
p 
™ "@Ґ=
6К3
0,€€€€€€€€€€€€€€€€€€€€€€€€€€€ј
Ъ р
O__inference_batch_normalization_4_layer_call_and_return_conditional_losses_4844ЬПРСТNҐK
DҐA
;К8
inputs,€€€€€€€€€€€€€€€€€€€€€€€€€€€ј
p
™ "@Ґ=
6К3
0,€€€€€€€€€€€€€€€€€€€€€€€€€€€ј
Ъ Ћ
O__inference_batch_normalization_4_layer_call_and_return_conditional_losses_4862xПРСТ<Ґ9
2Ґ/
)К&
inputs€€€€€€€€€ј
p 
™ ".Ґ+
$К!
0€€€€€€€€€ј
Ъ Ћ
O__inference_batch_normalization_4_layer_call_and_return_conditional_losses_4880xПРСТ<Ґ9
2Ґ/
)К&
inputs€€€€€€€€€ј
p
™ ".Ґ+
$К!
0€€€€€€€€€ј
Ъ »
4__inference_batch_normalization_4_layer_call_fn_4769ППРСТNҐK
DҐA
;К8
inputs,€€€€€€€€€€€€€€€€€€€€€€€€€€€ј
p 
™ "3К0,€€€€€€€€€€€€€€€€€€€€€€€€€€€ј»
4__inference_batch_normalization_4_layer_call_fn_4782ППРСТNҐK
DҐA
;К8
inputs,€€€€€€€€€€€€€€€€€€€€€€€€€€€ј
p
™ "3К0,€€€€€€€€€€€€€€€€€€€€€€€€€€€ј£
4__inference_batch_normalization_4_layer_call_fn_4795kПРСТ<Ґ9
2Ґ/
)К&
inputs€€€€€€€€€ј
p 
™ "!К€€€€€€€€€ј£
4__inference_batch_normalization_4_layer_call_fn_4808kПРСТ<Ґ9
2Ґ/
)К&
inputs€€€€€€€€€ј
p
™ "!К€€€€€€€€€јо
O__inference_batch_normalization_5_layer_call_and_return_conditional_losses_4979Ъ§•¶ІMҐJ
CҐ@
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€`
p 
™ "?Ґ<
5К2
0+€€€€€€€€€€€€€€€€€€€€€€€€€€€`
Ъ о
O__inference_batch_normalization_5_layer_call_and_return_conditional_losses_4997Ъ§•¶ІMҐJ
CҐ@
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€`
p
™ "?Ґ<
5К2
0+€€€€€€€€€€€€€€€€€€€€€€€€€€€`
Ъ …
O__inference_batch_normalization_5_layer_call_and_return_conditional_losses_5015v§•¶І;Ґ8
1Ґ.
(К%
inputs€€€€€€€€€  `
p 
™ "-Ґ*
#К 
0€€€€€€€€€  `
Ъ …
O__inference_batch_normalization_5_layer_call_and_return_conditional_losses_5033v§•¶І;Ґ8
1Ґ.
(К%
inputs€€€€€€€€€  `
p
™ "-Ґ*
#К 
0€€€€€€€€€  `
Ъ ∆
4__inference_batch_normalization_5_layer_call_fn_4922Н§•¶ІMҐJ
CҐ@
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€`
p 
™ "2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€`∆
4__inference_batch_normalization_5_layer_call_fn_4935Н§•¶ІMҐJ
CҐ@
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€`
p
™ "2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€`°
4__inference_batch_normalization_5_layer_call_fn_4948i§•¶І;Ґ8
1Ґ.
(К%
inputs€€€€€€€€€  `
p 
™ " К€€€€€€€€€  `°
4__inference_batch_normalization_5_layer_call_fn_4961i§•¶І;Ґ8
1Ґ.
(К%
inputs€€€€€€€€€  `
p
™ " К€€€€€€€€€  `о
O__inference_batch_normalization_6_layer_call_and_return_conditional_losses_5132ЪєЇїЉMҐJ
CҐ@
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€0
p 
™ "?Ґ<
5К2
0+€€€€€€€€€€€€€€€€€€€€€€€€€€€0
Ъ о
O__inference_batch_normalization_6_layer_call_and_return_conditional_losses_5150ЪєЇїЉMҐJ
CҐ@
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€0
p
™ "?Ґ<
5К2
0+€€€€€€€€€€€€€€€€€€€€€€€€€€€0
Ъ …
O__inference_batch_normalization_6_layer_call_and_return_conditional_losses_5168vєЇїЉ;Ґ8
1Ґ.
(К%
inputs€€€€€€€€€@@0
p 
™ "-Ґ*
#К 
0€€€€€€€€€@@0
Ъ …
O__inference_batch_normalization_6_layer_call_and_return_conditional_losses_5186vєЇїЉ;Ґ8
1Ґ.
(К%
inputs€€€€€€€€€@@0
p
™ "-Ґ*
#К 
0€€€€€€€€€@@0
Ъ ∆
4__inference_batch_normalization_6_layer_call_fn_5075НєЇїЉMҐJ
CҐ@
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€0
p 
™ "2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€0∆
4__inference_batch_normalization_6_layer_call_fn_5088НєЇїЉMҐJ
CҐ@
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€0
p
™ "2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€0°
4__inference_batch_normalization_6_layer_call_fn_5101iєЇїЉ;Ґ8
1Ґ.
(К%
inputs€€€€€€€€€@@0
p 
™ " К€€€€€€€€€@@0°
4__inference_batch_normalization_6_layer_call_fn_5114iєЇїЉ;Ґ8
1Ґ.
(К%
inputs€€€€€€€€€@@0
p
™ " К€€€€€€€€€@@0и
M__inference_batch_normalization_layer_call_and_return_conditional_losses_4134Ц1234MҐJ
CҐ@
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€
p 
™ "?Ґ<
5К2
0+€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ъ и
M__inference_batch_normalization_layer_call_and_return_conditional_losses_4152Ц1234MҐJ
CҐ@
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€
p
™ "?Ґ<
5К2
0+€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ъ «
M__inference_batch_normalization_layer_call_and_return_conditional_losses_4170v1234=Ґ:
3Ґ0
*К'
inputs€€€€€€€€€АА
p 
™ "/Ґ,
%К"
0€€€€€€€€€АА
Ъ «
M__inference_batch_normalization_layer_call_and_return_conditional_losses_4188v1234=Ґ:
3Ґ0
*К'
inputs€€€€€€€€€АА
p
™ "/Ґ,
%К"
0€€€€€€€€€АА
Ъ ј
2__inference_batch_normalization_layer_call_fn_4077Й1234MҐJ
CҐ@
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€
p 
™ "2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€ј
2__inference_batch_normalization_layer_call_fn_4090Й1234MҐJ
CҐ@
:К7
inputs+€€€€€€€€€€€€€€€€€€€€€€€€€€€
p
™ "2К/+€€€€€€€€€€€€€€€€€€€€€€€€€€€Я
2__inference_batch_normalization_layer_call_fn_4103i1234=Ґ:
3Ґ0
*К'
inputs€€€€€€€€€АА
p 
™ ""К€€€€€€€€€ААЯ
2__inference_batch_normalization_layer_call_fn_4116i1234=Ґ:
3Ґ0
*К'
inputs€€€€€€€€€АА
p
™ ""К€€€€€€€€€АА≤
B__inference_conv2d_1_layer_call_and_return_conditional_losses_4237lAB7Ґ4
-Ґ*
(К%
inputs€€€€€€€€€@@
™ "-Ґ*
#К 
0€€€€€€€€€@@ 
Ъ К
'__inference_conv2d_1_layer_call_fn_4227_AB7Ґ4
-Ґ*
(К%
inputs€€€€€€€€€@@
™ " К€€€€€€€€€@@ ≤
B__inference_conv2d_2_layer_call_and_return_conditional_losses_4410lXY7Ґ4
-Ґ*
(К%
inputs€€€€€€€€€   
™ "-Ґ*
#К 
0€€€€€€€€€  @
Ъ К
'__inference_conv2d_2_layer_call_fn_4400_XY7Ґ4
-Ґ*
(К%
inputs€€€€€€€€€   
™ " К€€€€€€€€€  @≥
B__inference_conv2d_3_layer_call_and_return_conditional_losses_4583mop7Ґ4
-Ґ*
(К%
inputs€€€€€€€€€@
™ ".Ґ+
$К!
0€€€€€€€€€А
Ъ Л
'__inference_conv2d_3_layer_call_fn_4573`op7Ґ4
-Ґ*
(К%
inputs€€€€€€€€€@
™ "!К€€€€€€€€€Аґ
B__inference_conv2d_4_layer_call_and_return_conditional_losses_4756pИЙ8Ґ5
.Ґ+
)К&
inputs€€€€€€€€€ј
™ ".Ґ+
$К!
0€€€€€€€€€ј
Ъ О
'__inference_conv2d_4_layer_call_fn_4746cИЙ8Ґ5
.Ґ+
)К&
inputs€€€€€€€€€ј
™ "!К€€€€€€€€€јµ
B__inference_conv2d_5_layer_call_and_return_conditional_losses_4909oЭЮ8Ґ5
.Ґ+
)К&
inputs€€€€€€€€€  а
™ "-Ґ*
#К 
0€€€€€€€€€  `
Ъ Н
'__inference_conv2d_5_layer_call_fn_4899bЭЮ8Ґ5
.Ґ+
)К&
inputs€€€€€€€€€  а
™ " К€€€€€€€€€  `і
B__inference_conv2d_6_layer_call_and_return_conditional_losses_5062n≤≥7Ґ4
-Ґ*
(К%
inputs€€€€€€€€€@@p
™ "-Ґ*
#К 
0€€€€€€€€€@@0
Ъ М
'__inference_conv2d_6_layer_call_fn_5052a≤≥7Ґ4
-Ґ*
(К%
inputs€€€€€€€€€@@p
™ " К€€€€€€€€€@@0Є
B__inference_conv2d_7_layer_call_and_return_conditional_losses_5216r∆«9Ґ6
/Ґ,
*К'
inputs€€€€€€€€€АА0
™ "/Ґ,
%К"
0€€€€€€€€€АА
Ъ Р
'__inference_conv2d_7_layer_call_fn_5205e∆«9Ґ6
/Ґ,
*К'
inputs€€€€€€€€€АА0
™ ""К€€€€€€€€€ААі
@__inference_conv2d_layer_call_and_return_conditional_losses_4064p*+9Ґ6
/Ґ,
*К'
inputs€€€€€€€€€АА
™ "/Ґ,
%К"
0€€€€€€€€€АА
Ъ М
%__inference_conv2d_layer_call_fn_4054c*+9Ґ6
/Ґ,
*К'
inputs€€€€€€€€€АА
™ ""К€€€€€€€€€ААм
I__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_4386ЮRҐO
HҐE
CК@
inputs4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ "HҐE
>К;
04€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ъ µ
I__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_4391h7Ґ4
-Ґ*
(К%
inputs€€€€€€€€€@@ 
™ "-Ґ*
#К 
0€€€€€€€€€   
Ъ ƒ
.__inference_max_pooling2d_1_layer_call_fn_4376СRҐO
HҐE
CК@
inputs4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ ";К84€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€Н
.__inference_max_pooling2d_1_layer_call_fn_4381[7Ґ4
-Ґ*
(К%
inputs€€€€€€€€€@@ 
™ " К€€€€€€€€€   м
I__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_4559ЮRҐO
HҐE
CК@
inputs4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ "HҐE
>К;
04€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ъ µ
I__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_4564h7Ґ4
-Ґ*
(К%
inputs€€€€€€€€€  @
™ "-Ґ*
#К 
0€€€€€€€€€@
Ъ ƒ
.__inference_max_pooling2d_2_layer_call_fn_4549СRҐO
HҐE
CК@
inputs4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ ";К84€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€Н
.__inference_max_pooling2d_2_layer_call_fn_4554[7Ґ4
-Ґ*
(К%
inputs€€€€€€€€€  @
™ " К€€€€€€€€€@м
I__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_4732ЮRҐO
HҐE
CК@
inputs4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ "HҐE
>К;
04€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ъ Ј
I__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_4737j8Ґ5
.Ґ+
)К&
inputs€€€€€€€€€А
™ ".Ґ+
$К!
0€€€€€€€€€А
Ъ ƒ
.__inference_max_pooling2d_3_layer_call_fn_4722СRҐO
HҐE
CК@
inputs4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ ";К84€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€П
.__inference_max_pooling2d_3_layer_call_fn_4727]8Ґ5
.Ґ+
)К&
inputs€€€€€€€€€А
™ "!К€€€€€€€€€Ак
G__inference_max_pooling2d_layer_call_and_return_conditional_losses_4213ЮRҐO
HҐE
CК@
inputs4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ "HҐE
>К;
04€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ъ µ
G__inference_max_pooling2d_layer_call_and_return_conditional_losses_4218j9Ґ6
/Ґ,
*К'
inputs€€€€€€€€€АА
™ "-Ґ*
#К 
0€€€€€€€€€@@
Ъ ¬
,__inference_max_pooling2d_layer_call_fn_4203СRҐO
HҐE
CК@
inputs4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ ";К84€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€Н
,__inference_max_pooling2d_layer_call_fn_4208]9Ґ6
/Ґ,
*К'
inputs€€€€€€€€€АА
™ " К€€€€€€€€€@@≠
A__inference_re_lu_1_layer_call_and_return_conditional_losses_4371h7Ґ4
-Ґ*
(К%
inputs€€€€€€€€€@@ 
™ "-Ґ*
#К 
0€€€€€€€€€@@ 
Ъ Е
&__inference_re_lu_1_layer_call_fn_4366[7Ґ4
-Ґ*
(К%
inputs€€€€€€€€€@@ 
™ " К€€€€€€€€€@@ ≠
A__inference_re_lu_2_layer_call_and_return_conditional_losses_4544h7Ґ4
-Ґ*
(К%
inputs€€€€€€€€€  @
™ "-Ґ*
#К 
0€€€€€€€€€  @
Ъ Е
&__inference_re_lu_2_layer_call_fn_4539[7Ґ4
-Ґ*
(К%
inputs€€€€€€€€€  @
™ " К€€€€€€€€€  @ѓ
A__inference_re_lu_3_layer_call_and_return_conditional_losses_4717j8Ґ5
.Ґ+
)К&
inputs€€€€€€€€€А
™ ".Ґ+
$К!
0€€€€€€€€€А
Ъ З
&__inference_re_lu_3_layer_call_fn_4712]8Ґ5
.Ґ+
)К&
inputs€€€€€€€€€А
™ "!К€€€€€€€€€Аѓ
A__inference_re_lu_4_layer_call_and_return_conditional_losses_4890j8Ґ5
.Ґ+
)К&
inputs€€€€€€€€€ј
™ ".Ґ+
$К!
0€€€€€€€€€ј
Ъ З
&__inference_re_lu_4_layer_call_fn_4885]8Ґ5
.Ґ+
)К&
inputs€€€€€€€€€ј
™ "!К€€€€€€€€€ј≠
A__inference_re_lu_5_layer_call_and_return_conditional_losses_5043h7Ґ4
-Ґ*
(К%
inputs€€€€€€€€€  `
™ "-Ґ*
#К 
0€€€€€€€€€  `
Ъ Е
&__inference_re_lu_5_layer_call_fn_5038[7Ґ4
-Ґ*
(К%
inputs€€€€€€€€€  `
™ " К€€€€€€€€€  `≠
A__inference_re_lu_6_layer_call_and_return_conditional_losses_5196h7Ґ4
-Ґ*
(К%
inputs€€€€€€€€€@@0
™ "-Ґ*
#К 
0€€€€€€€€€@@0
Ъ Е
&__inference_re_lu_6_layer_call_fn_5191[7Ґ4
-Ґ*
(К%
inputs€€€€€€€€€@@0
™ " К€€€€€€€€€@@0ѓ
?__inference_re_lu_layer_call_and_return_conditional_losses_4198l9Ґ6
/Ґ,
*К'
inputs€€€€€€€€€АА
™ "/Ґ,
%К"
0€€€€€€€€€АА
Ъ З
$__inference_re_lu_layer_call_fn_4193_9Ґ6
/Ґ,
*К'
inputs€€€€€€€€€АА
™ ""К€€€€€€€€€АА±
A__inference_reshape_layer_call_and_return_conditional_losses_5235l9Ґ6
/Ґ,
*К'
inputs€€€€€€€€€АА
™ "/Ґ,
%К"
0€€€€€€€€€АА
Ъ Й
&__inference_reshape_layer_call_fn_5221_9Ґ6
/Ґ,
*К'
inputs€€€€€€€€€АА
™ ""К€€€€€€€€€ААх
"__inference_signature_wrapper_3471ќH*+1234ABHIJKXY_`abopvwxy“ИЙПРСТ”ЭЮ§•¶І‘≤≥єЇїЉ’∆«EҐB
Ґ 
;™8
6
input_1+К(
input_1€€€€€€€€€АА";™8
6
reshape+К(
reshape€€€€€€€€€АА