use trading_lib;
use trading_lib::HistoryMetadata;

pub struct FileStorage{}

impl FileStorage {
    pub fn create() -> anyhow::Result<FileStorage> {
        Ok(FileStorage{})
    }
}

impl trading_lib::Storage for FileStorage {
    fn save_symbol_history(&mut self, name: &str, history: &Vec<trading_lib::HistoryStep>, metadata : &HistoryMetadata) -> anyhow::Result<()> {
        std::fs::create_dir_all(format!("history/{}", name))?;

        let file = std::fs::File::create(format!("history/{}/candlesticks.json", name))?;
        ::serde_json::to_writer(&file, history)?;

        let file = std::fs::File::create(format!("history/{}/metadata.json", name))?;
        ::serde_json::to_writer(&file, metadata)?;

        Ok(())
    }

    fn load_symbol_history(&mut self, name: &str) -> anyhow::Result<(Vec<trading_lib::HistoryStep>, HistoryMetadata)> {
        let file = std::fs::File::open(format!("{}/candlesticks.json", name))?;
        let history = ::serde_json::from_reader(&file)?;

        let file = std::fs::File::open(format!("{}/metadata.json", name))?;
        let metadata = ::serde_json::from_reader(&file)?;

        Ok((history, metadata))
    }
}