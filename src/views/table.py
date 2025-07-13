def print_table(df, title=None, max_col_width=30):
    """
    Imprime DataFrame no estilo MySQL/PostgreSQL CLI.
    
    Args:
        df (pd.DataFrame): DataFrame para imprimir
        title (str): Título opcional
        show_index (bool): Mostrar índice
        max_col_width (int): Largura máxima por coluna
    """
    if df.empty:
        print("Empty set (0 rows)")
        return
    
    # Preparar dados
    data = df.copy()
        
    # Calcular larguras
    col_widths = {}
    for col in data.columns:
        col_name_len = len(str(col))
        max_data_len = data[col].astype(str).str.len().max() if not data[col].empty else 0
        col_widths[col] = min(max(col_name_len, max_data_len) + 2, max_col_width)
    
    # Título
    if title:
        total_width = sum(col_widths.values()) + len(col_widths) * 3 + 1
        print("+" + "-" * (total_width - 2) + "+")
        print(f"| {title:^{total_width - 4}} |")
    
    # Linha superior
    line = "+"
    for width in col_widths.values():
        line += "-" * width + "+"
    print(line)
    
    # Cabeçalho
    header = "|"
    for col, width in col_widths.items():
        header += f" {str(col):<{width-1}}|"
    print(header)
    
    # Separador
    print(line)
    
    # Dados
    for _, row in data.iterrows():
        row_str = "|"
        for col, width in col_widths.items():
            value = str(row[col])
            if len(value) > width - 1:
                value = value[:width-4] + "..."
            row_str += f" {value:<{width-1}}|"
        print(row_str)
    
    # Linha final
    print(line)