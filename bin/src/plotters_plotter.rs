use trading_lib;
use plotters::prelude::*;

pub struct PlottersPlotter{}

impl PlottersPlotter {
    pub fn create() -> anyhow::Result<PlottersPlotter> {
        Ok(PlottersPlotter{})
    }
}

impl trading_lib::Plotter for PlottersPlotter {
    fn plot_lines(&mut self, y_points_list : &Vec<(String, Vec<f32>)>, title : &str, filename : &str) -> anyhow::Result<()> {
        let png_filename = format!("{}.png", filename);
        let root_area =
            BitMapBackend::new(&png_filename, (1920, 1080)).into_drawing_area();
        // let png_filename = format!("{}.svg", filename);
        // let root_area =
        //     SVGBackend::new(&png_filename, (1920, 1080)).into_drawing_area();
        root_area.fill(&WHITE)?;

        let root_area = root_area.titled(title, ("sans-serif", 18))?;
        // let (upper, _lower) = root_area.split_vertically(512);

        let x_axis = (0.0..y_points_list[0].1.len() as f32).step(1.0);
        let mut min_y = f32::INFINITY;
        let mut max_y = f32::NEG_INFINITY;
        for (_name, y_points) in y_points_list {
            for i in 0..y_points.len() {
                if y_points[i] < min_y {
                    min_y = y_points[i];
                }
                if y_points[i] > max_y {
                    max_y = y_points[i];
                }
            }
        }

        let mut cc = ChartBuilder::on(&root_area)
            .margin(5)
            .set_all_label_area_size(50)
            .build_cartesian_2d(0.0..y_points_list[0].1.len() as f32, min_y..max_y)?;

        cc.configure_mesh()
            .x_labels(20)
            .y_labels(10)
            // .disable_mesh()
            // .x_label_formatter(&|v| format!("{:.1}", v))
            // .y_label_formatter(&|v| format!("{:.1}", v))
            .draw()?;

        for i in 0..y_points_list.len() {
            let (label, y_points) = &y_points_list[i];
            let color = PlottersPlotter::get_color(i);
            cc.draw_series(LineSeries::new(x_axis.values().map(|i| (i, y_points[i.floor() as usize])), &color))?
                .label(label)
                .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &color));
        }

        cc.configure_series_labels().border_style(&BLACK).draw()?;

        Ok(())
    }
}

impl PlottersPlotter {
    fn get_color(i: usize) -> RGBColor {
        let color = match i {
            0 => RED,
            1 => GREEN,
            2 => BLUE,
            3 => YELLOW,
            4 => CYAN,
            _ => MAGENTA
        };
        color
    }
}
