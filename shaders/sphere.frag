#version 450

// Освещение по модели Блинна-Фонга (лекция 4):
// L = ambient * albedo + сумма по источникам (albedo + specular * (n·h)^alpha) * (n·l) * E

layout(location = 0) in vec3 in_world_pos;
layout(location = 1) in vec3 in_world_normal;

layout(location = 0) out vec4 out_color;

layout(std140, set = 0, binding = 0) uniform SceneUniforms {
	mat4 view_projection;
	vec3 camera_position;
	vec3 directional_light_direction; // куда светит направленный источник
	vec3 directional_light_color;     // цвет * интенсивность
	vec3 ambient_light;               // цвет * интенсивность
	uint point_light_count;
	uint spot_light_count;
} scene_uniforms;

layout(std140, set = 1, binding = 0) uniform ModelUniform {
	mat4 model;
	vec3 albedo;   // цвет материала
	vec3 specular; // цвет блика
	float alpha;   // коэффициент блеска
} model_uniforms;

// Источники света как в лекции (слайд 32). В свободные 4 байта после vec3
// добавлены интенсивность и углы прожектора, размер структур от этого не меняется
struct PointLight {
	vec3 position;
	float intensity;
	vec3 color;
};

struct SpotLight {
	vec3 position;
	float intensity;
	vec3 color;
	float inner_angle; // косинус внутреннего угла
	vec3 direction;
	float outer_angle; // косинус внешнего угла
};

// Storage-буферы: размер массивов в шейдере не задаётся
layout(std430, set = 0, binding = 1) readonly buffer PointLights {
	PointLight point_lights[];
};

layout(std430, set = 0, binding = 2) readonly buffer SpotLights {
	SpotLight spot_lights[];
};

// Вклад одного источника: n - нормаль, v - на камеру, l - на источник, light - освещённость E
vec3 blinnPhong(vec3 n, vec3 v, vec3 l, vec3 light) {
	vec3 h = normalize(v + l); // половинчатый вектор
	float n_dot_l = max(dot(n, l), 0.0);
	float n_dot_h = max(dot(n, h), 0.0);
	return (model_uniforms.albedo + model_uniforms.specular * pow(n_dot_h, model_uniforms.alpha)) * n_dot_l * light;
}

void main() {
	vec3 n = normalize(in_world_normal);
	vec3 v = normalize(scene_uniforms.camera_position - in_world_pos);

	// Рассеянный свет
	vec3 color = scene_uniforms.ambient_light * model_uniforms.albedo;

	// Направленный источник: лучи параллельны, затухания нет
	color += blinnPhong(n, v, normalize(-scene_uniforms.directional_light_direction),
	                    scene_uniforms.directional_light_color);

	// Точечные источники: затухание по закону обратных квадратов
	for (uint i = 0; i < scene_uniforms.point_light_count; ++i) {
		vec3 to_light = point_lights[i].position - in_world_pos;
		float dist2 = max(dot(to_light, to_light), 0.0001);
		vec3 l = to_light / sqrt(dist2);
		vec3 light = point_lights[i].color * point_lights[i].intensity / dist2;
		color += blinnPhong(n, v, l, light);
	}

	// Прожекторы: обратные квадраты + мягкий край между внутренним и внешним углом
	for (uint i = 0; i < scene_uniforms.spot_light_count; ++i) {
		vec3 to_light = spot_lights[i].position - in_world_pos;
		float dist2 = max(dot(to_light, to_light), 0.0001);
		vec3 l = to_light / sqrt(dist2);

		// cos угла между осью прожектора и лучом к точке
		float cos_theta = dot(-l, spot_lights[i].direction);
		// 1 внутри внутреннего конуса, 0 за внешним, между ними плавный переход
		float edge = clamp((cos_theta - spot_lights[i].outer_angle) /
		                   max(spot_lights[i].inner_angle - spot_lights[i].outer_angle, 0.0001), 0.0, 1.0);

		vec3 light = spot_lights[i].color * spot_lights[i].intensity * edge / dist2;
		color += blinnPhong(n, v, l, light);
	}

	out_color = vec4(color, 1.0);
}
