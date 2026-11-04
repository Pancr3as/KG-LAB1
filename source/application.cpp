// Лабораторная работа №2: камера и освещение (вариант 5: сфера)
// Сделано базовое задание и доп. задания 1, 2, 3:
//   1 - два набора дескрипторов: set 0 - сцена (uniform-буфер + storage-буферы источников), set 1 - модель
//   2 - массив точечных источников в storage-буфере, затухание по закону обратных квадратов
//   3 - массив прожекторов в storage-буфере, затухание по закону обратных квадратов, мягкие края

#include "application.hpp"
#include "shaders_spirv.hpp" // скомпилированные шейдеры

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <vector>

#include <imgui.h>

namespace application {

namespace {

constexpr float PI = 3.14159265f;
constexpr float DEG = PI / 180.0f;

// ---------------- Матрицы 4x4 (хранятся по столбцам, как в GLSL) - из лабы 1

struct Mat4 {
	float m[4][4] = {}; // m[столбец][строка]
};

Mat4 identity() {
	Mat4 r;
	r.m[0][0] = r.m[1][1] = r.m[2][2] = r.m[3][3] = 1.0f;
	return r;
}

Mat4 operator*(const Mat4& a, const Mat4& b) {
	Mat4 r;
	for (int c = 0; c < 4; ++c)
		for (int row = 0; row < 4; ++row)
			for (int k = 0; k < 4; ++k)
				r.m[c][row] += a.m[k][row] * b.m[c][k];
	return r;
}

Mat4 translate(float x, float y, float z) {
	Mat4 r = identity();
	r.m[3][0] = x; r.m[3][1] = y; r.m[3][2] = z;
	return r;
}

Mat4 scale(float x, float y, float z) {
	Mat4 r = identity();
	r.m[0][0] = x; r.m[1][1] = y; r.m[2][2] = z;
	return r;
}

Mat4 rotateX(float a) {
	Mat4 r = identity();
	r.m[1][1] = std::cos(a);  r.m[1][2] = std::sin(a);
	r.m[2][1] = -std::sin(a); r.m[2][2] = std::cos(a);
	return r;
}

Mat4 rotateY(float a) {
	Mat4 r = identity();
	r.m[0][0] = std::cos(a); r.m[0][2] = -std::sin(a);
	r.m[2][0] = std::sin(a); r.m[2][2] = std::cos(a);
	return r;
}

Mat4 rotateZ(float a) {
	Mat4 r = identity();
	r.m[0][0] = std::cos(a);  r.m[0][1] = std::sin(a);
	r.m[1][0] = -std::sin(a); r.m[1][1] = std::cos(a);
	return r;
}

// Перспективная проекция (во Vulkan ось Y направлена вниз, глубина от 0 до 1)
Mat4 perspective(float fov, float aspect, float n, float f) {
	const float t = std::tan(fov / 2.0f);
	Mat4 r;
	r.m[0][0] = 1.0f / (aspect * t);
	r.m[1][1] = -1.0f / t;
	r.m[2][2] = f / (n - f);
	r.m[2][3] = -1.0f;
	r.m[3][2] = n * f / (n - f);
	return r;
}

// Ортографическая проекция: видимая область шириной 2w и высотой 2h
Mat4 orthographic(float w, float h, float n, float f) {
	Mat4 r;
	r.m[0][0] = 1.0f / w;
	r.m[1][1] = -1.0f / h;
	r.m[2][2] = 1.0f / (n - f);
	r.m[3][2] = n / (n - f);
	r.m[3][3] = 1.0f;
	return r;
}

// Делает вектор единичной длины (если он не нулевой)
void normalize3(const float in[3], float out[3]) {
	const float len = std::sqrt(in[0] * in[0] + in[1] * in[1] + in[2] * in[2]);
	const float k = len > 1e-6f ? 1.0f / len : 0.0f;
	out[0] = in[0] * k; out[1] = in[1] * k; out[2] = in[2] * k;
}

// ---------------- Данные для видеокарты (раскладка как в шейдерах)

// Вершина: позиция и нормаль
struct Vertex {
	float pos[3];
	float normal[3];
};

// set = 0, binding = 0, std140. vec3 выравнивается на 16 байт, поэтому после
// каждого vec3 стоит заполнитель, кроме ambient_light: за ним сразу идёт uint
struct SceneUniforms {
	Mat4 view_projection;                  // 0
	float camera_position[3];  float pad0; // 64
	float light_direction[3];  float pad1; // 80  (directional_light_direction)
	float light_color[3];      float pad2; // 96  (directional_light_color)
	float ambient_light[3];                // 112
	uint32_t point_light_count;            // 124
	uint32_t spot_light_count;             // 128
	uint32_t pad3[3];
};
static_assert(offsetof(SceneUniforms, point_light_count) == 124, "std140");
static_assert(offsetof(SceneUniforms, spot_light_count) == 128, "std140");

// set = 1, binding = 0, std140
struct ModelUniform {
	Mat4 model;                     // 0
	float albedo[3];   float pad;   // 64
	float specular[3]; float alpha; // 80, 92
};
static_assert(sizeof(ModelUniform) == 96, "std140");

// Элементы storage-буферов (std430)
struct PointLightGPU {
	float position[3]; float intensity;
	float color[3];    float pad;
};
static_assert(sizeof(PointLightGPU) == 32, "std430");

struct SpotLightGPU {
	float position[3];  float intensity;
	float color[3];     float inner_angle; // косинус
	float direction[3]; float outer_angle; // косинус
};
static_assert(sizeof(SpotLightGPU) == 48, "std430");

// ---------------- Параметры из интерфейса

// Камера
float cam_position[3] = { 0.0f, 3.0f, 9.0f };
float cam_yaw = 0.0f;     // поворот вокруг оси Y, градусы (0 - смотрим вдоль -Z)
float cam_pitch = -18.0f; // наклон вверх/вниз, градусы
float cam_fov = 60.0f;
bool perspective_on = true; // из лабы 1

// Движение сферы по кругу (из лабы 1)
bool animate = true;
float speed = 1.0f;
float radius = 2.0f;
float anim_time = 0.0f;
double last_time = 0.0;

// Рассеянный свет
float ambient_color[3] = { 1.0f, 1.0f, 1.0f };
float ambient_intensity = 0.08f;

// Направленный источник
float sun_direction[3] = { -0.4f, -1.0f, -0.5f };
float sun_color[3] = { 1.0f, 0.95f, 0.85f };
float sun_intensity = 0.3f;

// Точечный источник (доп. 2)
struct PointLight {
	float position[3];
	float color[3];
	float intensity;
};

// Прожектор (доп. 3)
struct SpotLight {
	float position[3];
	float direction[3];
	float color[3];
	float intensity;
	float inner_angle; // градусы
	float outer_angle; // градусы
};

std::vector<PointLight> point_lights = {
	{ { 2.5f, 0.5f, 2.5f }, { 1.0f, 0.3f, 0.1f }, 6.0f },
	{ { -3.0f, 1.5f, 0.5f }, { 0.2f, 0.5f, 1.0f }, 6.0f },
};

std::vector<SpotLight> spot_lights = {
	{ { 0.0f, 5.0f, 2.0f }, { 0.0f, -1.0f, -0.4f }, { 1.0f, 0.95f, 0.8f }, 30.0f, 15.0f, 25.0f },
};

// ---------------- Объекты сцены

struct Mesh {
	VkBuffer vertex_buffer, index_buffer;
	VmaAllocation vertex_memory, index_memory;
	uint32_t index_count;
};

struct Object {
	const char* name;
	Mesh* mesh;
	float position[3];
	float rotation[3]; // градусы
	float scaling[3];
	float albedo[3];   // цвет материала
	float specular[3]; // цвет блика
	float alpha;       // коэффициент блеска
	bool moves;        // двигается ли по кругу (как сфера в лабе 1)

	// У каждого объекта свой uniform-буфер и свой набор дескрипторов (set = 1)
	VkBuffer buffer;
	VmaAllocation memory;
	void* mapped;
	VkDescriptorSet set;
};

Mesh sphere_mesh, floor_mesh;

std::vector<Object> objects = {
	{ "Floor",    &floor_mesh,  { 0, -1, 0 },  { 0, 0, 0 }, { 1, 1, 1 },          { 0.6f, 0.6f, 0.6f }, { 0.2f, 0.2f, 0.2f }, 8.0f,   false },
	{ "Sphere",   &sphere_mesh, { 0, 0, 0 },   { 0, 0, 0 }, { 1, 1, 1 },          { 0.9f, 0.3f, 0.3f }, { 1.0f, 1.0f, 1.0f }, 64.0f,  true },
	{ "Sphere 2", &sphere_mesh, { -3, 0, -2 }, { 0, 0, 0 }, { 0.8f, 0.8f, 0.8f }, { 0.3f, 0.6f, 1.0f }, { 1.0f, 1.0f, 1.0f }, 128.0f, false },
	{ "Sphere 3", &sphere_mesh, { 3, -0.4f, -2 }, { 0, 0, 0 }, { 0.6f, 0.6f, 0.6f }, { 0.4f, 0.9f, 0.4f }, { 0.3f, 0.3f, 0.3f }, 16.0f, false },
};

// ---------------- Объекты Vulkan

VkDescriptorSetLayout scene_layout, model_layout;
VkDescriptorPool descriptor_pool;
VkDescriptorSet scene_set;
VkPipelineLayout pipeline_layout;
VkPipeline pipeline;

VkBuffer scene_buffer;
VmaAllocation scene_memory;
void* scene_mapped;

// Storage-буферы источников: размер растёт, если источников стало больше
VkBuffer point_buffer = VK_NULL_HANDLE, spot_buffer = VK_NULL_HANDLE;
VmaAllocation point_memory, spot_memory;
void* point_mapped;
void* spot_mapped;
size_t point_capacity = 0, spot_capacity = 0; // сколько элементов помещается

// Создаёт буфер в памяти, доступной процессору. Если data не nullptr - копирует туда данные.
// Возвращает указатель, по которому можно писать в буфер
void* createBuffer(const void* data, size_t size, VkBufferUsageFlags usage,
                   VkBuffer& buffer, VmaAllocation& memory) {
	const VkBufferCreateInfo info = {
		.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
		.size = size,
		.usage = usage,
	};
	const VmaAllocationCreateInfo alloc = {
		.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT,
		.usage = VMA_MEMORY_USAGE_AUTO,
	};
	VmaAllocationInfo result = {};
	auto& ctx = graphics::internal::context;
	vmaCreateBuffer(ctx.allocator, &info, &alloc, &buffer, &memory, &result);
	if (result.pMappedData != nullptr && data != nullptr) {
		std::memcpy(result.pMappedData, data, size);
		vmaFlushAllocation(ctx.allocator, memory, 0, size);
	}
	return result.pMappedData;
}

// Копирует данные в уже созданный буфер
void writeBuffer(void* mapped, VmaAllocation memory, const void* data, size_t size) {
	std::memcpy(mapped, data, size);
	vmaFlushAllocation(graphics::internal::context.allocator, memory, 0, size);
}

// Привязывает буфер к binding в наборе дескрипторов
void writeDescriptor(VkDescriptorSet set, uint32_t binding, VkDescriptorType type, VkBuffer buffer) {
	const VkDescriptorBufferInfo buffer_info = { buffer, 0, VK_WHOLE_SIZE };
	const VkWriteDescriptorSet write = {
		.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
		.dstSet = set,
		.dstBinding = binding,
		.descriptorCount = 1,
		.descriptorType = type,
		.pBufferInfo = &buffer_info,
	};
	vkUpdateDescriptorSets(graphics::internal::context.device, 1, &write, 0, nullptr);
}

// Если в storage-буфер не помещается count элементов, пересоздаёт его в два раза больше
// и обновляет дескриптор в наборе сцены
void reserveStorage(VkBuffer& buffer, VmaAllocation& memory, void*& mapped, size_t& capacity,
                    size_t count, size_t element_size, uint32_t binding) {
	if (buffer != VK_NULL_HANDLE && count <= capacity) return;

	auto& ctx = graphics::internal::context;
	if (buffer != VK_NULL_HANDLE) {
		vmaDestroyBuffer(ctx.allocator, buffer, memory);
	}
	capacity = std::max<size_t>(capacity * 2, 8);
	while (capacity < count) capacity *= 2;

	mapped = createBuffer(nullptr, capacity * element_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, buffer, memory);
	if (scene_set != VK_NULL_HANDLE) {
		writeDescriptor(scene_set, binding, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, buffer);
	}
}

void createMesh(const std::vector<Vertex>& vertices, const std::vector<uint32_t>& indices, Mesh& mesh) {
	mesh.index_count = uint32_t(indices.size());
	createBuffer(vertices.data(), vertices.size() * sizeof(Vertex),
	             VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, mesh.vertex_buffer, mesh.vertex_memory);
	createBuffer(indices.data(), indices.size() * sizeof(uint32_t),
	             VK_BUFFER_USAGE_INDEX_BUFFER_BIT, mesh.index_buffer, mesh.index_memory);
}

// Сфера радиуса 1 из лабы 1: 2 полюса + 7 параллелей по 14 точек = 100 вершин.
// Нормаль считается процедурно: у сферы с центром в нуле она совпадает с позицией вершины
void createSphere() {
	const int stacks = 8, slices = 14;
	std::vector<Vertex> vertices;
	std::vector<uint32_t> indices;

	auto add = [&](float x, float y, float z) {
		const float p[3] = { x, y, z };
		Vertex vertex = { { x, y, z }, {} };
		normalize3(p, vertex.normal);
		vertices.push_back(vertex);
	};

	add(0, 1, 0); // верхний полюс
	for (int i = 1; i < stacks; ++i) {
		const float phi = PI * i / stacks;
		for (int j = 0; j < slices; ++j) {
			const float theta = 2 * PI * j / slices;
			add(std::sin(phi) * std::cos(theta), std::cos(phi), std::sin(phi) * std::sin(theta));
		}
	}
	add(0, -1, 0); // нижний полюс

	const uint32_t top = 0, bottom = uint32_t(vertices.size() - 1);
	auto v = [&](int i, int j) { return uint32_t(1 + (i - 1) * slices + j % slices); };

	for (int j = 0; j < slices; ++j) {
		indices.insert(indices.end(), { top, v(1, j), v(1, j + 1) });
		for (int i = 1; i < stacks - 1; ++i) {
			indices.insert(indices.end(), { v(i, j), v(i + 1, j), v(i, j + 1) });
			indices.insert(indices.end(), { v(i, j + 1), v(i + 1, j), v(i + 1, j + 1) });
		}
		indices.insert(indices.end(), { v(stacks - 1, j), bottom, v(stacks - 1, j + 1) });
	}

	createMesh(vertices, indices, sphere_mesh);
}

// Пол: квадрат 20x20 в плоскости y = 0, нормаль смотрит вверх
void createFloor() {
	const float s = 10.0f;
	const std::vector<Vertex> vertices = {
		{ { -s, 0, -s }, { 0, 1, 0 } },
		{ { -s, 0,  s }, { 0, 1, 0 } },
		{ {  s, 0,  s }, { 0, 1, 0 } },
		{ {  s, 0, -s }, { 0, 1, 0 } },
	};
	createMesh(vertices, { 0, 1, 2, 0, 2, 3 }, floor_mesh);
}

// Доп. 1: два набора дескрипторов
void createDescriptors() {
	auto& ctx = graphics::internal::context;

	// set = 0 (сцена): binding 0 - uniform-буфер сцены, 1 - точечные источники, 2 - прожекторы
	const VkDescriptorSetLayoutBinding scene_bindings[] = {
		{ 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT },
		{ 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT },
		{ 2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT },
	};
	const VkDescriptorSetLayoutCreateInfo scene_info = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
		.bindingCount = 3,
		.pBindings = scene_bindings,
	};
	vkCreateDescriptorSetLayout(ctx.device, &scene_info, nullptr, &scene_layout);

	// set = 1 (модель): binding 0 - uniform-буфер с матрицей модели и материалом
	const VkDescriptorSetLayoutBinding model_binding =
		{ 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT };
	const VkDescriptorSetLayoutCreateInfo model_info = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
		.bindingCount = 1,
		.pBindings = &model_binding,
	};
	vkCreateDescriptorSetLayout(ctx.device, &model_info, nullptr, &model_layout);

	// Пул: 1 набор сцены + по одному набору на каждый объект
	const uint32_t count = uint32_t(objects.size());
	const VkDescriptorPoolSize sizes[] = {
		{ VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1 + count },
		{ VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2 },
	};
	const VkDescriptorPoolCreateInfo pool_info = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
		.maxSets = 1 + count,
		.poolSizeCount = 2,
		.pPoolSizes = sizes,
	};
	vkCreateDescriptorPool(ctx.device, &pool_info, nullptr, &descriptor_pool);

	// Набор сцены
	VkDescriptorSetAllocateInfo alloc = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
		.descriptorPool = descriptor_pool,
		.descriptorSetCount = 1,
		.pSetLayouts = &scene_layout,
	};
	vkAllocateDescriptorSets(ctx.device, &alloc, &scene_set);

	scene_mapped = createBuffer(nullptr, sizeof(SceneUniforms), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
	                            scene_buffer, scene_memory);
	writeDescriptor(scene_set, 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, scene_buffer);
	reserveStorage(point_buffer, point_memory, point_mapped, point_capacity, 1, sizeof(PointLightGPU), 1);
	reserveStorage(spot_buffer, spot_memory, spot_mapped, spot_capacity, 1, sizeof(SpotLightGPU), 2);

	// Наборы моделей: у каждого объекта свой uniform-буфер
	alloc.pSetLayouts = &model_layout;
	for (Object& object : objects) {
		vkAllocateDescriptorSets(ctx.device, &alloc, &object.set);
		object.mapped = createBuffer(nullptr, sizeof(ModelUniform), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
		                             object.buffer, object.memory);
		writeDescriptor(object.set, 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, object.buffer);
	}
}

VkShaderModule createShader(const uint32_t* code, size_t size) {
	const VkShaderModuleCreateInfo info = {
		.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
		.codeSize = size,
		.pCode = code,
	};
	VkShaderModule module;
	vkCreateShaderModule(graphics::internal::context.device, &info, nullptr, &module);
	return module;
}

void createPipeline() {
	auto& ctx = graphics::internal::context;

	// Конвейер видит два набора дескрипторов: set 0 - сцена, set 1 - модель
	const VkDescriptorSetLayout set_layouts[] = { scene_layout, model_layout };
	const VkPipelineLayoutCreateInfo layout_info = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
		.setLayoutCount = 2,
		.pSetLayouts = set_layouts,
	};
	vkCreatePipelineLayout(ctx.device, &layout_info, nullptr, &pipeline_layout);

	VkShaderModule vert = createShader(vert_code, sizeof(vert_code));
	VkShaderModule frag = createShader(frag_code, sizeof(frag_code));
	const VkPipelineShaderStageCreateInfo stages[] = {
		{ .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
		  .stage = VK_SHADER_STAGE_VERTEX_BIT, .module = vert, .pName = "main" },
		{ .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
		  .stage = VK_SHADER_STAGE_FRAGMENT_BIT, .module = frag, .pName = "main" },
	};

	// Вершина: позиция (location 0) и нормаль (location 1)
	const VkVertexInputBindingDescription binding = { 0, sizeof(Vertex), VK_VERTEX_INPUT_RATE_VERTEX };
	const VkVertexInputAttributeDescription attributes[] = {
		{ 0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, pos) },
		{ 1, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, normal) },
	};
	const VkPipelineVertexInputStateCreateInfo vertex_input = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
		.vertexBindingDescriptionCount = 1,
		.pVertexBindingDescriptions = &binding,
		.vertexAttributeDescriptionCount = 2,
		.pVertexAttributeDescriptions = attributes,
	};
	const VkPipelineInputAssemblyStateCreateInfo input_assembly = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
		.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
	};
	const VkPipelineViewportStateCreateInfo viewport = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
		.viewportCount = 1,
		.scissorCount = 1,
	};
	const VkPipelineRasterizationStateCreateInfo raster = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
		.polygonMode = VK_POLYGON_MODE_FILL,
		.cullMode = VK_CULL_MODE_NONE,
		.lineWidth = 1.0f,
	};
	const VkPipelineMultisampleStateCreateInfo multisample = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
		.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
	};
	const VkPipelineDepthStencilStateCreateInfo depth = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
		.depthTestEnable = VK_TRUE,
		.depthWriteEnable = VK_TRUE,
		.depthCompareOp = VK_COMPARE_OP_LESS,
	};
	const VkPipelineColorBlendAttachmentState blend_attachment = {
		.colorWriteMask = 0xF, // R, G, B, A
	};
	const VkPipelineColorBlendStateCreateInfo blend = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
		.attachmentCount = 1,
		.pAttachments = &blend_attachment,
	};
	const VkDynamicState dynamic_states[] = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
	const VkPipelineDynamicStateCreateInfo dynamic = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
		.dynamicStateCount = 2,
		.pDynamicStates = dynamic_states,
	};

	const VkGraphicsPipelineCreateInfo info = {
		.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
		.stageCount = 2,
		.pStages = stages,
		.pVertexInputState = &vertex_input,
		.pInputAssemblyState = &input_assembly,
		.pViewportState = &viewport,
		.pRasterizationState = &raster,
		.pMultisampleState = &multisample,
		.pDepthStencilState = &depth,
		.pColorBlendState = &blend,
		.pDynamicState = &dynamic,
		.layout = pipeline_layout,
		.renderPass = ctx.render_pass,
	};
	vkCreateGraphicsPipelines(ctx.device, VK_NULL_HANDLE, 1, &info, nullptr, &pipeline);

	vkDestroyShaderModule(ctx.device, vert, nullptr);
	vkDestroyShaderModule(ctx.device, frag, nullptr);
}

// Направление прожектора по умолчанию - туда, куда смотрит камера
void cameraForward(float out[3]) {
	const float yaw = cam_yaw * DEG, pitch = cam_pitch * DEG;
	out[0] = std::sin(yaw) * std::cos(pitch);
	out[1] = std::sin(pitch);
	out[2] = -std::cos(yaw) * std::cos(pitch);
}

} // namespace

bool initialize() {
	createSphere();
	createFloor();
	createDescriptors();
	createPipeline();
	return true;
}

void shutdown() {
	auto& ctx = graphics::internal::context;
	vkQueueWaitIdle(ctx.graphics_queue);

	for (Object& object : objects) {
		vmaDestroyBuffer(ctx.allocator, object.buffer, object.memory);
	}
	vmaDestroyBuffer(ctx.allocator, scene_buffer, scene_memory);
	vmaDestroyBuffer(ctx.allocator, point_buffer, point_memory);
	vmaDestroyBuffer(ctx.allocator, spot_buffer, spot_memory);
	for (Mesh* mesh : { &sphere_mesh, &floor_mesh }) {
		vmaDestroyBuffer(ctx.allocator, mesh->vertex_buffer, mesh->vertex_memory);
		vmaDestroyBuffer(ctx.allocator, mesh->index_buffer, mesh->index_memory);
	}

	vkDestroyPipeline(ctx.device, pipeline, nullptr);
	vkDestroyPipelineLayout(ctx.device, pipeline_layout, nullptr);
	vkDestroyDescriptorPool(ctx.device, descriptor_pool, nullptr); // наборы удаляются вместе с пулом
	vkDestroyDescriptorSetLayout(ctx.device, scene_layout, nullptr);
	vkDestroyDescriptorSetLayout(ctx.device, model_layout, nullptr);
}

void update(double time) {
	if (animate) {
		anim_time += float(time - last_time) * speed;
	}
	last_time = time;

	ImGui::SetNextWindowPos(ImVec2(10, 10), ImGuiCond_FirstUseEver);
	ImGui::SetNextWindowSize(ImVec2(360, 0), ImGuiCond_FirstUseEver);
	ImGui::Begin("Lab 2");

	if (ImGui::CollapsingHeader("Camera", ImGuiTreeNodeFlags_DefaultOpen)) {
		ImGui::DragFloat3("Position", cam_position, 0.05f);
		ImGui::SliderFloat("Yaw", &cam_yaw, -180.0f, 180.0f);
		ImGui::SliderFloat("Pitch", &cam_pitch, -89.0f, 89.0f);
		ImGui::SliderFloat("FOV", &cam_fov, 20.0f, 120.0f);
		ImGui::Checkbox("Perspective", &perspective_on);
	}

	if (ImGui::CollapsingHeader("Animation")) {
		ImGui::Checkbox("Animate", &animate);
		ImGui::SliderFloat("Speed", &speed, 0.0f, 5.0f);
		ImGui::SliderFloat("Radius", &radius, 0.0f, 3.0f);
	}

	// Параметры материала каждого объекта
	if (ImGui::CollapsingHeader("Objects")) {
		for (size_t i = 0; i < objects.size(); ++i) {
			Object& object = objects[i];
			ImGui::PushID(int(i));
			if (ImGui::TreeNode(object.name)) {
				ImGui::DragFloat3("Position", object.position, 0.05f);
				ImGui::DragFloat3("Rotation", object.rotation, 1.0f);
				ImGui::DragFloat3("Scale", object.scaling, 0.01f);
				ImGui::ColorEdit3("Albedo", object.albedo);
				ImGui::ColorEdit3("Specular", object.specular);
				ImGui::SliderFloat("Shininess", &object.alpha, 1.0f, 256.0f);
				ImGui::TreePop();
			}
			ImGui::PopID();
		}
	}

	if (ImGui::CollapsingHeader("Ambient light")) {
		ImGui::ColorEdit3("Color##ambient", ambient_color);
		ImGui::SliderFloat("Intensity##ambient", &ambient_intensity, 0.0f, 1.0f);
	}

	if (ImGui::CollapsingHeader("Directional light", ImGuiTreeNodeFlags_DefaultOpen)) {
		ImGui::DragFloat3("Direction##sun", sun_direction, 0.01f, -1.0f, 1.0f);
		ImGui::ColorEdit3("Color##sun", sun_color);
		ImGui::SliderFloat("Intensity##sun", &sun_intensity, 0.0f, 3.0f);
	}

	// Доп. 2: точечные источники - добавление, удаление, параметры
	if (ImGui::CollapsingHeader("Point lights")) {
		if (ImGui::Button("Add point light")) {
			point_lights.push_back({ { 0.0f, 2.0f, 2.0f }, { 1.0f, 1.0f, 1.0f }, 5.0f });
		}
		int remove = -1; // удаляем после цикла, чтобы не сломать перебор массива
		for (size_t i = 0; i < point_lights.size(); ++i) {
			PointLight& light = point_lights[i];
			ImGui::PushID(int(i));
			if (ImGui::TreeNode("point", "Point light %d", int(i) + 1)) {
				ImGui::DragFloat3("Position", light.position, 0.05f);
				ImGui::ColorEdit3("Color", light.color);
				ImGui::DragFloat("Intensity", &light.intensity, 0.1f, 0.0f, 200.0f);
				if (ImGui::Button("Remove")) remove = int(i);
				ImGui::TreePop();
			}
			ImGui::PopID();
		}
		if (remove >= 0) point_lights.erase(point_lights.begin() + remove);
	}

	// Доп. 3: прожекторы - добавление, удаление, параметры
	if (ImGui::CollapsingHeader("Spot lights")) {
		if (ImGui::Button("Add spot light")) {
			// новый прожектор ставим в позицию камеры и направляем туда же, куда она смотрит
			SpotLight light = { {}, {}, { 1.0f, 1.0f, 1.0f }, 20.0f, 15.0f, 25.0f };
			std::memcpy(light.position, cam_position, sizeof(light.position));
			cameraForward(light.direction);
			spot_lights.push_back(light);
		}
		int remove = -1;
		for (size_t i = 0; i < spot_lights.size(); ++i) {
			SpotLight& light = spot_lights[i];
			ImGui::PushID(int(i) + 1000);
			if (ImGui::TreeNode("spot", "Spot light %d", int(i) + 1)) {
				ImGui::DragFloat3("Position", light.position, 0.05f);
				ImGui::DragFloat3("Direction", light.direction, 0.01f, -1.0f, 1.0f);
				ImGui::ColorEdit3("Color", light.color);
				ImGui::DragFloat("Intensity", &light.intensity, 0.1f, 0.0f, 500.0f);
				ImGui::SliderFloat("Inner angle", &light.inner_angle, 0.0f, 89.0f);
				ImGui::SliderFloat("Outer angle", &light.outer_angle, 0.0f, 89.0f);
				// внешний угол не меньше внутреннего (если больше - край пятна мягкий)
				light.outer_angle = std::max(light.outer_angle, light.inner_angle);
				if (ImGui::Button("Remove")) remove = int(i);
				ImGui::TreePop();
			}
			ImGui::PopID();
		}
		if (remove >= 0) spot_lights.erase(spot_lights.begin() + remove);
	}

	ImGui::End();
}

void render(const graphics::internal::FrameData& fd) {
	auto& ctx = graphics::internal::context;
	const VkCommandBuffer cmd = fd.command_buffer;
	const VkExtent2D extent = ctx.swapchain_extent;
	const float aspect = float(extent.width) / float(extent.height);

	// ---- Камера. Матрица вида - обратная к матрице "модели" камеры (лекция 4):
	// камера = T(позиция) * Ry(-yaw) * Rx(pitch), значит вид = Rx(-pitch) * Ry(yaw) * T(-позиция)
	const Mat4 view = rotateX(-cam_pitch * DEG) * rotateY(cam_yaw * DEG) *
		translate(-cam_position[0], -cam_position[1], -cam_position[2]);

	const Mat4 projection = perspective_on
		? perspective(cam_fov * DEG, aspect, 0.1f, 100.0f)
		: orthographic(5 * aspect, 5, 0.1f, 100.0f);

	// ---- Данные сцены (set 0, binding 0)
	SceneUniforms scene = {};
	scene.view_projection = projection * view;
	std::memcpy(scene.camera_position, cam_position, sizeof(scene.camera_position));
	normalize3(sun_direction, scene.light_direction);
	for (int k = 0; k < 3; ++k) {
		scene.light_color[k] = sun_color[k] * sun_intensity;
		scene.ambient_light[k] = ambient_color[k] * ambient_intensity;
	}
	scene.point_light_count = uint32_t(point_lights.size());
	scene.spot_light_count = uint32_t(spot_lights.size());

	// ---- Массивы источников (set 0, binding 1 и 2)
	std::vector<PointLightGPU> points;
	for (const PointLight& light : point_lights) {
		PointLightGPU p = {};
		std::memcpy(p.position, light.position, sizeof(p.position));
		std::memcpy(p.color, light.color, sizeof(p.color));
		p.intensity = light.intensity;
		points.push_back(p);
	}

	std::vector<SpotLightGPU> spots;
	for (const SpotLight& light : spot_lights) {
		SpotLightGPU s = {};
		std::memcpy(s.position, light.position, sizeof(s.position));
		std::memcpy(s.color, light.color, sizeof(s.color));
		normalize3(light.direction, s.direction);
		s.intensity = light.intensity;
		// в шейдер передаём косинусы углов, чтобы не считать их для каждого пикселя
		s.inner_angle = std::cos(light.inner_angle * DEG);
		s.outer_angle = std::cos(light.outer_angle * DEG);
		spots.push_back(s);
	}

	// К этому моменту видеокарта уже закончила прошлый кадр (prepare() ждёт его),
	// поэтому буферы можно перезаписывать и пересоздавать
	reserveStorage(point_buffer, point_memory, point_mapped, point_capacity, points.size(), sizeof(PointLightGPU), 1);
	reserveStorage(spot_buffer, spot_memory, spot_mapped, spot_capacity, spots.size(), sizeof(SpotLightGPU), 2);

	writeBuffer(scene_mapped, scene_memory, &scene, sizeof(scene));
	if (!points.empty()) writeBuffer(point_mapped, point_memory, points.data(), points.size() * sizeof(PointLightGPU));
	if (!spots.empty()) writeBuffer(spot_mapped, spot_memory, spots.data(), spots.size() * sizeof(SpotLightGPU));

	// ---- Данные моделей (set 1): матрица модели и материал
	for (Object& object : objects) {
		float x = object.position[0], y = object.position[1], z = object.position[2];
		float spin = 0.0f;
		if (object.moves) {
			// движение по кругу и вращение вокруг своей оси, как в лабе 1
			x += radius * std::cos(anim_time);
			z += radius * std::sin(anim_time);
			spin = anim_time;
		}

		ModelUniform model = {};
		model.model = translate(x, y, z) *
			rotateZ(object.rotation[2] * DEG) * rotateY(object.rotation[1] * DEG + spin) * rotateX(object.rotation[0] * DEG) *
			scale(object.scaling[0], object.scaling[1], object.scaling[2]);
		std::memcpy(model.albedo, object.albedo, sizeof(model.albedo));
		std::memcpy(model.specular, object.specular, sizeof(model.specular));
		model.alpha = object.alpha;
		writeBuffer(object.mapped, object.memory, &model, sizeof(model));
	}

	// ---- Запись команд
	const VkCommandBufferBeginInfo begin = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
	vkBeginCommandBuffer(cmd, &begin);

	VkClearValue clear[2] = {};
	clear[0].color = { { 0.02f, 0.02f, 0.03f, 1.0f } };
	clear[1].depthStencil = { 1.0f, 0 };
	const VkRenderPassBeginInfo pass = {
		.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
		.renderPass = ctx.render_pass,
		.framebuffer = fd.framebuffer,
		.renderArea = { .extent = extent },
		.clearValueCount = 2,
		.pClearValues = clear,
	};
	vkCmdBeginRenderPass(cmd, &pass, VK_SUBPASS_CONTENTS_INLINE);

	const VkViewport viewport = { 0, 0, float(extent.width), float(extent.height), 0, 1 };
	const VkRect2D scissor = { { 0, 0 }, extent };
	vkCmdSetViewport(cmd, 0, 1, &viewport);
	vkCmdSetScissor(cmd, 0, 1, &scissor);

	vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);

	// Набор сцены (set 0) привязываем один раз - он общий для всех объектов
	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout, 0, 1, &scene_set, 0, nullptr);

	for (const Object& object : objects) {
		// Набор модели (set 1) у каждого объекта свой
		vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout, 1, 1, &object.set, 0, nullptr);
		const VkDeviceSize offset = 0;
		vkCmdBindVertexBuffers(cmd, 0, 1, &object.mesh->vertex_buffer, &offset);
		vkCmdBindIndexBuffer(cmd, object.mesh->index_buffer, 0, VK_INDEX_TYPE_UINT32);
		vkCmdDrawIndexed(cmd, object.mesh->index_count, 1, 0, 0, 0);
	}

	vkCmdEndRenderPass(cmd);
	vkEndCommandBuffer(cmd);
}

} // namespace application
