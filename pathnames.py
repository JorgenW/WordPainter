'''
Created 2016 at NTNU.

This file contains the dictionary for the paths of the NNs and their color mean.
'''

pathnames_list = {
	"ImageNet": {
		"net_fn"   	: '../caffe/models/bvlc_googlenet/deploy.prototxt',
		"param_fn" 	: '../caffe/models/bvlc_googlenet/bvlc_googlenet.caffemodel',
		"mean" 		: [104.0, 116.0, 122.0]
	},
	"MIT Places": {
		"net_fn"   	: '../caffe/models/googlenet_places205/deploy_places205.protxt',
		"param_fn" 	: '../caffe/models/googlenet_places205/googlelet_places205_train_iter_2400000.caffemodel',
		"mean" 		: [104.0, 116.0, 122.0]
	},
	"Cars": {
		"net_fn"   	: '../caffe/models/googlenet_cars/deploy.prototxt',
		"param_fn" 	: '../caffe/models/googlenet_cars/googlenet_finetune_web_car_iter_10000.caffemodel',
		"mean" 		: [104.0, 116.0, 122.0]
	},
	"Flowers": {
		"net_fn"   	: '../caffe/models/flowers/deploy.prototxt',
		"param_fn" 	: '../caffe/models/flowers/MYNET_iter_125000.caffemodel',
		"mean" 		: [104.0, 116.0, 122.0]
	},
	"Fjordnet": {
		"net_fn"   	: '../caffe/models/fjordnet/deploy.prototxt',
		"param_fn" 	: '../caffe/models/fjordnet/MYNET_iter_200000.caffemodel',
		"mean" 		: [104.0, 116.0, 122.0]
	},
	"Pigsnet": {
		"net_fn"   	: '../caffe/models/pigsnet/deploy.prototxt',
		"param_fn" 	: '../caffe/models/pigsnet/MYNET_iter_145000.caffemodel',
		"mean" 		: [104.0, 116.0, 122.0]
	},
	"Horsenet": {
		"net_fn"   	: '../caffe/models/horsenet/deploy.prototxt',
		"param_fn" 	: '../caffe/models/horsenet/NEW_HORSENET_iter_340000.caffemodel',
		"mean" 		: [121.0, 119.0, 105.0]
	},
	"caffenet": {
		"net_fn"   	: '../caffe/models/bvlc_reference_caffenet/deploy.prototxt',
		"param_fn" 	: '../caffe/models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel',
		"mean" 		: [104.0, 116.0, 122.0]
	},
	"notredame": {
		"net_fn"   	: '../caffe/models/notredame/deploy.prototxt',
		"param_fn" 	: '../caffe/models/notredame/NOTREDAME_iter_374111.caffemodel',
		"mean" 		: [119.0, 119.0, 119.0]
	},
	"notredame2": {
		"net_fn"   	: '../caffe/models/notredame2/deploy.prototxt',
		"param_fn" 	: '../caffe/models/notredame2/NOTREDAME2_iter_400000.caffemodel',
		"mean" 		: [119.0, 119.0, 119.0]
	},
	"notredame3": {
		"net_fn"   	: '../caffe/models/notredame3/deploy.prototxt',
		"param_fn" 	: '../caffe/models/notredame3/NOTREDAME2_iter_400000.caffemodel',
		"mean" 		: [119.0, 119.0, 119.0]
	},
	"helieverest": {
		"net_fn"   	: '../caffe/models/helieverest/deploy.prototxt',
		"param_fn" 	: '../caffe/models/helieverest/HELIEVEREST_iter_200000.caffemodel',
		"mean" 		: [119.0, 119.0, 119.0]
	},
	"helieverest2": {
		"net_fn"   	: '../caffe/models/helieverest2/deploy.prototxt',
		"param_fn" 	: '../caffe/models/helieverest2/HELIEVEREST2_iter_200000.caffemodel',
		"mean" 		: [119.0, 119.0, 119.0]
	},
	"helieverest3": {
		"net_fn"   	: '../caffe/models/helieverest3/deploy.prototxt',
		"param_fn" 	: '../caffe/models/helieverest3/HELIEVEREST3_iter_200000.caffemodel',
		"mean" 		: [119.0, 119.0, 119.0]
	},
	"africanet": {
		"net_fn"   	: '../caffe/models/africanet/deploy.prototxt',
		"param_fn" 	: '../caffe/models/africanet/AFRICANET_iter_400000.caffemodel',
		"mean" 		: [119.0, 119.0, 119.0]
	}
}
