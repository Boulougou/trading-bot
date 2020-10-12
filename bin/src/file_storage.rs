use trading_lib;

pub struct FileStorage{}

impl FileStorage {
    pub fn create() -> anyhow::Result<FileStorage> {
        Ok(FileStorage{})
    }
}

impl trading_lib::Storage for FileStorage {
    fn save_symbol_history(&mut self, name: &str, history: &Vec<trading_lib::HistoryStep>) -> anyhow::Result<()> {
        let file = std::fs::File::create(format!("{}.json", name))?;

        ::serde_json::to_writer(&file, history)?;
        Ok(())
    }

    fn load_symbol_history(&mut self, name: &str) -> anyhow::Result<Vec<trading_lib::HistoryStep>> {
        let file = std::fs::File::open(format!("{}.json", name))?;
        let history = ::serde_json::from_reader(&file)?;
        Ok(history)
    }
}