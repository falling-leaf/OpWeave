/*
 * @Author: Anonymous anonymous@example.com
 * @Date: 2023-05-07 14:10:23
 * @LastEditors: Anonymous anonymous@example.com
 * @LastEditTime: 2023-07-06 17:22:23
 * @FilePath: /postgres-DB4AI/src/udf/env.h
 * @Description: This is the default setting. Please set `customMade`, open koroFileHeader to view the configuration: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
 */
#pragma once
#ifndef _ENV_H_
#define _ENV_H_


#include "c10/core/DeviceType.h"
#include "c10/core/Device.h"
#include <torch/torch.h>
#include <torch/script.h>
#include <opencv2/core/mat.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

#ifdef __cplusplus
extern "C" {
#endif

#ifdef LOG
#undef LOG
#endif

#include "postgres.h"
#include "fmgr.h"

#ifdef __cplusplus
}
#endif

//#include <opencv/cv.h>

// to use gettext, dgettext and other functions from libintl.h
//#define ENABLE_NLS




#endif
