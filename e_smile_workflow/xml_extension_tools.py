import streamlit as st
import copy


st.subheader("EXTENSION 编辑器")
if "ext_list" not in st.session_state:
    st.session_state.ext_list = []
def add_tag(tag, idx, name):
    st.session_state.ext_list.append({
        "tag": tag,
        "index": idx,
        "name": name
    })

col1, col2, col3 = st.columns(3)
with col1:
    if st.button(":heavy_plus_sign: 添加 <a>（原子）"):
        add_tag("a", 0, "R")
with col2:
    if st.button(":heavy_plus_sign: 添加 <r>（环）"):
        add_tag("r", 0, "Ar")
with col3:
    if st.button(":heavy_plus_sign: 添加 <dum>（连接点）"):
        add_tag("a", 0, "<dum>")
# 显示当前 EXTENSION
for i, item in enumerate(st.session_state.ext_list):
    c1, c2, c3, c4 = st.columns([2, 2, 3, 1])
    with c1:
        tag = st.selectbox(
            "Tag", ["a", "r", "c"],
            index=["a","r","c"].index(item["tag"]),
            key=f"tag_{i}"
        )
    with c2:
        idx = st.number_input(
            "Index", value=item["index"], key=f"idx_{i}"
        )
    with c3:
        name = st.text_input(
            "Group Name", value=item["name"], key=f"name_{i}"
        )
    with c4:
        if st.button(":wastebasket:", key=f"del_{i}"):
            st.session_state.ext_list.pop(i)
            st.experimental_rerun()
    st.session_state.ext_list[i] = {
        "tag": tag,
        "index": idx,
        "name": name
    }

def ext_list_to_xml(ext_list):
    xml = ""
    for item in ext_list:
        xml += f"<{item['tag']}>{item['index']}:{item['name']}</{item['tag']}>"
    return xml

xml_out = ext_list_to_xml(st.session_state.ext_list)
st.code(xml_out, language="xml")
