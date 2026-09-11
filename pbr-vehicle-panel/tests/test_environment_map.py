import numpy as np

from pbr_vehicle_standalone.environment_map import (
    EnvironmentMap,
    cube_face_rotations,
    textured_environment_sphere,
)


def test_constant_cubemap_produces_constant_diffuse_irradiance():
    color = np.asarray([0.2, 0.4, 0.8], dtype=np.float32)
    faces = np.broadcast_to(color, (6, 32, 32, 3)).copy()
    environment = EnvironmentMap.from_faces(faces)
    normals = np.asarray(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [-1.0, 0.0, 0.0]],
        dtype=np.float32,
    )
    assert np.allclose(environment.diffuse(normals), color[None, :], atol=1e-3)


def test_cubemap_face_centers_sample_their_source_faces():
    faces = np.zeros((6, 8, 8, 3), dtype=np.float32)
    expected = []
    for index in range(6):
        color = np.eye(3, dtype=np.float32)[index % 3]
        faces[index] = color
        expected.append(color)
    environment = EnvironmentMap.from_faces(faces)
    assert np.allclose(
        environment.sample(cube_face_rotations()[:, :, 2]),
        np.asarray(expected),
    )


def test_reflection_sampling_changes_with_view_direction():
    faces = np.zeros((6, 8, 8, 3), dtype=np.float32)
    faces[0] = [1.0, 0.0, 0.0]
    faces[1] = [0.0, 1.0, 0.0]
    environment = EnvironmentMap.from_faces(faces)
    normals = np.asarray([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32)
    views = np.asarray([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]], dtype=np.float32)
    reflected = environment.reflection(normals, views, 0.0)
    assert np.allclose(reflected[0], [1.0, 0.0, 0.0])
    assert np.allclose(reflected[1], [0.0, 1.0, 0.0])


def test_environment_preview_sphere_contains_exportable_equirectangular_texture():
    faces = np.ones((6, 8, 8, 3), dtype=np.float32) * np.asarray([0.2, 0.4, 0.8])
    environment = EnvironmentMap.from_faces(faces)
    image = environment.to_equirectangular()
    sphere = textured_environment_sphere(environment, latitudes=8, longitudes=16)

    assert image.shape == (16, 32, 3)
    assert np.allclose(image, [0.2, 0.4, 0.8], atol=1e-5)
    assert sphere.visual.uv.shape == (len(sphere.vertices), 2)
    assert len(sphere.export(file_type="glb")) > 0
