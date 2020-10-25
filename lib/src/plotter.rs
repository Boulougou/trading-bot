#[cfg(test)]
use mockall::{automock};

#[cfg_attr(test, automock)]
pub trait Plotter {
    fn plot_lines(&mut self, y_points_list : &Vec<(String, Vec<f32>)>, title : &str, filename : &str) -> anyhow::Result<()>;
}
