import os
import json
import glob
import yaml

def extract_dhbb_corpus(dhbb_text_dir="dhbb/text", output_json="dhbb_corpus.json"):
    corpus = {}
    
    # Vai buscar todos os ficheiros .md ou .txt dentro da pasta text
    files = glob.glob(os.path.join(dhbb_text_dir, "*.*"))
    print(f"Encontrados {len(files)} ficheiros. A processar...")
    
    for filepath in files:
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # O ficheiro começa com '---', tem o YAML, e fecha com outro '---'
            parts = content.split('---')
            if len(parts) >= 3:
                yaml_metadata = parts[1]
                text_content = '---'.join(parts[2:]).strip()
                
                # Ler os metadados
                metadata = yaml.safe_load(yaml_metadata)
                
                # Tentar obter o título/identificador da entidade
                title = metadata.get('title', '')
                if not title and 'name' in metadata:
                    title = metadata['name']
                    
                if title:
                    # Guardamos apenas os primeiros 1000 caracteres para não sobrecarregar a memória
                    corpus[title] = text_content[:1000] 
        except Exception as e:
            continue # Ignora ficheiros com erro
            
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(corpus, f, ensure_ascii=False, indent=2)
        
    print(f"Sucesso! {len(corpus)} entidades extraídas para {output_json}")

if __name__ == "__main__":
    extract_dhbb_corpus()