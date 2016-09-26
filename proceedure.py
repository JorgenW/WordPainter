'''
Created 2016 at NTNU.

This file contains the dictionary for the generation proceedure.
test1 				: name of test
"inputImage"		: path to image to generate on
"word"				: guide-word, require images of that word. If No Guide, write "No Guide"
"numberOfImages"	:
"database"			:
"iterations"		: 
"octaves"			: 5,
"octaveScale"		: 1.4,
"model"				: Name of NN. See pathnames.py
"layer"				: Name of layer for output, for example "inception_4c/output",
"jitter"			: 15,
"stepSize"			: 1.5,
"mean"				: [104.0, 116.0, 122.0]
'''


proceedures = {
	"test1": {
		"inputImage": "exampleImages/white.jpeg",
		"word"				: "No Guide",
		"numberOfImages"	: 10,
		"database"			: "Flickr",
		"iterations"		: 10,
		"octaves"			: 8,
		"octaveScale"		: 1.4,
		"model"				: "helieverest",
		"layer"				: "inception_4a/outnew",
		"jitter"			: 32,
		"stepSize"			: 1.5,
		"mean"				: [119.0, 119.0, 119.0],
		"focus"				: None
	},
	"test2": {
		"inputImage": "exampleImages/white.jpeg",
		"word"				: "No Guide",
		"numberOfImages"	: 10,
		"database"			: "Flickr",
		"iterations"		: 10,
		"octaves"			: 8,
		"octaveScale"		: 1.4,
		"model"				: "helieverest",
		"layer"				: "inception_4b/outnew",
		"jitter"			: 32,
		"stepSize"			: 1.5,
		"mean"				: [119.0, 119.0, 119.0],
		"focus"				: None
	},
	"test3": {
		"inputImage": "exampleImages/white.jpeg",
		"word"				: "No Guide",
		"numberOfImages"	: 10,
		"database"			: "Flickr",
		"iterations"		: 10,
		"octaves"			: 8,
		"octaveScale"		: 1.4,
		"model"				: "helieverest",
		"layer"				: "inception_4c/outnew",
		"jitter"			: 32,
		"stepSize"			: 1.5,
		"mean"				: [119.0, 119.0, 119.0],
		"focus"				: None
	},
	"test4": {
		"inputImage": "exampleImages/white.jpeg",
		"word"				: "No Guide",
		"numberOfImages"	: 10,
		"database"			: "Flickr",
		"iterations"		: 10,
		"octaves"			: 8,
		"octaveScale"		: 1.4,
		"model"				: "helieverest",
		"layer"				: "inception_4d/outnew",
		"jitter"			: 32,
		"stepSize"			: 1.5,
		"mean"				: [119.0, 119.0, 119.0],
		"focus"				: None
	},
	"test5": {
		"inputImage": "exampleImages/white.jpeg",
		"word"				: "No Guide",
		"numberOfImages"	: 10,
		"database"			: "Flickr",
		"iterations"		: 10,
		"octaves"			: 8,
		"octaveScale"		: 1.4,
		"model"				: "helieverest",
		"layer"				: "inception_5a/outnew",
		"jitter"			: 32,
		"stepSize"			: 1.5,
		"mean"				: [119.0, 119.0, 119.0],
		"focus"				: None
	},
	"test6": {
		"inputImage": "exampleImages/white.jpeg",
		"word"				: "No Guide",
		"numberOfImages"	: 10,
		"database"			: "Flickr",
		"iterations"		: 10,
		"octaves"			: 8,
		"octaveScale"		: 1.4,
		"model"				: "helieverest",
		"layer"				: "inception_5b/outnew",
		"jitter"			: 32,
		"stepSize"			: 1.5,
		"mean"				: [119.0, 119.0, 119.0],
		"focus"				: None
	}
}
