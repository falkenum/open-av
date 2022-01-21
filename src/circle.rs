struct Circle {
    vertices: Vec<Vertex>,
    indices: Vec<u16>,
}

impl Circle {
    fn new_centered(num_edges: u16, color_at_edge_idx: fn(u16, u16) -> [f32; 3]) -> Circle {
        let default_radius = 0.5;

        let mut vertices = vec![Vertex::get_center([0.5, 0.5, 0.5])];
        for i in 0..num_edges {
            let twopi = 4.0 * libm::asinf(1.0);
            let angle = (i as f32/ num_edges as f32) * twopi;
            vertices.push(Vertex::new([
                default_radius*libm::cosf(angle),
                default_radius*libm::sinf(angle),
                0.0,
            ], color_at_edge_idx(i, num_edges)));
        }
        let mut indices = vec![];
        for i in 0..num_edges {
            indices.push(0);
            indices.push(1 + i);
            indices.push(1 + (i+1) % num_edges)
        }

        Circle {
            vertices: vertices,
            indices: indices
        }
    }
}