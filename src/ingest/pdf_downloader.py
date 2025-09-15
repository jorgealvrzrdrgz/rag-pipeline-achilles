import requests
from pathlib import Path


class PdfDownloader:
    def __init__(self):
        self.data_dir = Path(__file__).parent.parent.parent / "data"
        self.data_dir.mkdir(exist_ok=True)

    def download(self, pdf_url: str) -> str:
        """
        Descarga un PDF desde una URL y lo guarda en la carpeta data.
        Solo descarga si el archivo no existe previamente.
        
        Args:
            pdf_url: URL del PDF a descargar
            
        Returns:
            str: Ruta del archivo (descargado o existente)
        """
        filename = pdf_url.split("/")[-1]
        if not filename.endswith('.pdf'):
            filename += '.pdf'
        file_path = self.data_dir / filename
        
        # Verificar si el archivo ya existe
        if file_path.exists():
            print(f"PDF ya existe, omitiendo descarga: {file_path}")
            return str(file_path)
        
        # Descargar solo si no existe
        response = requests.get(pdf_url)
        response.raise_for_status()  # Lanza excepción si hay error HTTP

        with open(file_path, 'wb') as f:
            f.write(response.content)
        
        print(f"PDF descargado: {file_path}")
        return str(file_path)
