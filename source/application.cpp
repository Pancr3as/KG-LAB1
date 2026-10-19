// Лабораторная работа №1, вариант 5: сфера (~100 вершин)
// Доп. задания: 1 - смена проекции, 2 - позиция/поворот/масштаб,
// 3 - движение по кругу (пауза, скорость, радиус), 4 - цвет, 5 - цвета вершин.

#include "application.hpp"
#include "shaders_spirv.hpp" // скомпилированные шейдеры

#include <cmath>
#include <cstddef>
#include <cstring>
#include <vector>

#include <imgui.h>

namespace application {

namespace {

constexpr float PI = 3.14159265f;

// ---------------- Матрицы 4x4 (хранятся по столбцам, как в GLSL)

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

// ---------------- Данные

struct Vertex {
	float pos[3];
	float color[3];
};

struct Push {
	Mat4 mvp;
	float color[4];
	int use_vertex_colors;
};

// Параметры из интерфейса
bool perspective_on = true;          // доп. 1
float position[3] = { 0, 0, 0 };     // доп. 2
float rotation[3] = { 0, 0, 0 };     // в градусах
float scaling[3] = { 1, 1, 1 };
bool animate = true;                 // доп. 3
float speed = 1.0f;
float radius = 2.0f;
float color[3] = { 1, 1, 1 };        // доп. 4
bool use_vertex_colors = true;       // доп. 5

float anim_time = 0.0f;
double last_time = 0.0;

VkPipelineLayout pipeline_layout;
VkPipeline pipeline;
VkBuffer vertex_buffer, index_buffer;
VmaAllocation vertex_memory, index_memory;
uint32_t index_count;

// Создаёт буфер и копирует в него данные
void createBuffer(const void* data, size_t size, VkBufferUsageFlags usage,
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
	if (result.pMappedData == nullptr) return;
	std::memcpy(result.pMappedData, data, size);
	vmaFlushAllocation(ctx.allocator, memory, 0, size);
}

// Сфера радиуса 1: 2 полюса + 7 параллелей по 14 точек = 100 вершин
void createSphere() {
	const int stacks = 8, slices = 14;
	std::vector<Vertex> vertices;
	std::vector<uint32_t> indices;

	// цвет вершины = её координаты, переведённые из [-1, 1] в [0, 1]
	auto add = [&](float x, float y, float z) {
		vertices.push_back({ { x, y, z }, { x * 0.5f + 0.5f, y * 0.5f + 0.5f, z * 0.5f + 0.5f } });
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

	index_count = uint32_t(indices.size());
	createBuffer(vertices.data(), vertices.size() * sizeof(Vertex),
	             VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, vertex_buffer, vertex_memory);
	createBuffer(indices.data(), indices.size() * sizeof(uint32_t),
	             VK_BUFFER_USAGE_INDEX_BUFFER_BIT, index_buffer, index_memory);
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

	const VkPushConstantRange push_range = { VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(Push) };
	const VkPipelineLayoutCreateInfo layout_info = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
		.pushConstantRangeCount = 1,
		.pPushConstantRanges = &push_range,
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

	// Вершина: позиция (location 0) и цвет (location 1)
	const VkVertexInputBindingDescription binding = { 0, sizeof(Vertex), VK_VERTEX_INPUT_RATE_VERTEX };
	const VkVertexInputAttributeDescription attributes[] = {
		{ 0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, pos) },
		{ 1, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, color) },
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

} // namespace

bool initialize() {
	createPipeline();
	createSphere();
	return true;
}

void shutdown() {
	auto& ctx = graphics::internal::context;
	vkQueueWaitIdle(ctx.graphics_queue);
	vmaDestroyBuffer(ctx.allocator, vertex_buffer, vertex_memory);
	vmaDestroyBuffer(ctx.allocator, index_buffer, index_memory);
	vkDestroyPipeline(ctx.device, pipeline, nullptr);
	vkDestroyPipelineLayout(ctx.device, pipeline_layout, nullptr);
}

void update(double time) {
	if (animate) {
		anim_time += float(time - last_time) * speed;
	}
	last_time = time;

	ImGui::SetNextWindowSize(ImVec2(350, 0), ImGuiCond_FirstUseEver);
	ImGui::Begin("Sphere");
	ImGui::Checkbox("Perspective", &perspective_on);           // доп. 1
	ImGui::DragFloat3("Position", position, 0.05f);            // доп. 2
	ImGui::DragFloat3("Rotation", rotation, 1.0f);
	ImGui::DragFloat3("Scale", scaling, 0.01f);
	ImGui::Checkbox("Animate", &animate);                      // доп. 3
	ImGui::SliderFloat("Speed", &speed, 0.0f, 5.0f);
	ImGui::SliderFloat("Radius", &radius, 0.0f, 3.0f);
	ImGui::ColorEdit3("Color", color);                         // доп. 4
	ImGui::Checkbox("Vertex colors", &use_vertex_colors);      // доп. 5
	ImGui::End();
}

void render(const graphics::internal::FrameData& fd) {
	auto& ctx = graphics::internal::context;
	const VkCommandBuffer cmd = fd.command_buffer;
	const VkExtent2D extent = ctx.swapchain_extent;
	const float aspect = float(extent.width) / float(extent.height);
	const float deg = PI / 180.0f;

	// Модель: движение по кругу + вращение вокруг своей оси, затем трансформации из интерфейса
	const Mat4 model =
		translate(position[0] + radius * std::cos(anim_time), position[1], position[2] + radius * std::sin(anim_time)) *
		rotateZ(rotation[2] * deg) * rotateY(rotation[1] * deg + anim_time) * rotateX(rotation[0] * deg) *
		scale(scaling[0], scaling[1], scaling[2]);

	// Камера: отодвинута на 8 и смотрит немного сверху
	const Mat4 view = translate(0, 0, -8) * rotateX(25 * deg);

	const Mat4 projection = perspective_on
		? perspective(60 * deg, aspect, 0.1f, 100.0f)
		: orthographic(4 * aspect, 4, 0.1f, 100.0f);

	Push push = {};
	push.mvp = projection * view * model;
	push.color[0] = color[0];
	push.color[1] = color[1];
	push.color[2] = color[2];
	push.use_vertex_colors = use_vertex_colors;

	const VkCommandBufferBeginInfo begin = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
	vkBeginCommandBuffer(cmd, &begin);

	VkClearValue clear[2] = {};
	clear[0].color = { { 0.0f, 0.0f, 0.0f, 1.0f } };
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
	vkCmdPushConstants(cmd, pipeline_layout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(push), &push);
	const VkDeviceSize offset = 0;
	vkCmdBindVertexBuffers(cmd, 0, 1, &vertex_buffer, &offset);
	vkCmdBindIndexBuffer(cmd, index_buffer, 0, VK_INDEX_TYPE_UINT32);
	vkCmdDrawIndexed(cmd, index_count, 1, 0, 0, 0);

	vkCmdEndRenderPass(cmd);
	vkEndCommandBuffer(cmd);
}

} // namespace application
