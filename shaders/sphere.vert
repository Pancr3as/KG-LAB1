#version 450

layout(location = 0) in vec3 in_pos;
layout(location = 1) in vec3 in_color;

layout(push_constant) uniform Push {
	mat4 mvp;              // projection * view * model
	vec4 color;            // цвет из интерфейса
	int use_vertex_colors; // 1 - умножать на цвет вершины
} pc;

layout(location = 0) out vec3 out_color;

void main() {
	gl_Position = pc.mvp * vec4(in_pos, 1.0);
	out_color = pc.use_vertex_colors != 0 ? in_color * pc.color.rgb : pc.color.rgb;
}
