#version 450

// Атрибуты вершины
layout(location = 0) in vec3 in_pos;
layout(location = 1) in vec3 in_normal;

// В фрагментный шейдер передаём позицию и нормаль в мировых координатах
layout(location = 0) out vec3 out_world_pos;
layout(location = 1) out vec3 out_world_normal;

// set = 0: общие данные сцены (лекция 4, слайд 31)
layout(std140, set = 0, binding = 0) uniform SceneUniforms {
	mat4 view_projection;
	vec3 camera_position;
	vec3 directional_light_direction;
	vec3 directional_light_color;
	vec3 ambient_light;
	uint point_light_count;
	uint spot_light_count;
} scene_uniforms;

// set = 1: данные модели
layout(std140, set = 1, binding = 0) uniform ModelUniform {
	mat4 model;
	vec3 albedo;
	vec3 specular;
	float alpha;
} model_uniforms;

void main() {
	vec4 world_pos = model_uniforms.model * vec4(in_pos, 1.0);

	// Нормаль переводим матрицей, обратной и транспонированной к матрице модели,
	// иначе при растяжении объекта нормали перестанут быть перпендикулярны поверхности
	mat3 normal_matrix = transpose(inverse(mat3(model_uniforms.model)));

	out_world_pos = world_pos.xyz;
	out_world_normal = normal_matrix * in_normal;
	gl_Position = scene_uniforms.view_projection * world_pos;
}
