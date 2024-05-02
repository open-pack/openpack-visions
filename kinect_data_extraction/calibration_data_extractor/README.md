# Calibration Data Extractor for k4a

Extract the camera calibration for K4a device from the recordings (i.e., `*.mkv`)

## How to ues?

- Build this package

```bash
mkdir build
cd build
cmake ..
make
cd ..
```

- Download a sample mkv file.

```bash
make download
```

- Run `k4a_calibration_data_extractor`

```bash
make run
```

or

```bash
mkdir -p ./outputs
./build/k4a_calibration_data_extractor ../../samples/sample.mkv ./outputs/camera_calibration_params.json
```

Then, you will get the results as a JSON file like this.

```json
{
    "color_camera_calibration": {
        "extrinsics": {
            "rotation": [
                0.9999960660934448,
                -0.0025176280178129673,
                0.0012580311158671975,
                0.0023888947907835245,
                0.9956113696098328,
                0.09355393797159195,
                -0.001488044043071568,
                -0.09355056285858154,
                0.9956133961677551
            ],
            "translation": [
                -31.909168243408203,
                -2.089625358581543,
                4.0456719398498535
            ]
        },
        "intrinsics": {
            "parameter_count": 14,
            "parameters": {}
        },
        ...
    },
    ...
}
```
