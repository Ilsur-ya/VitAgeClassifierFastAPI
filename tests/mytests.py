
def test_old_man(client):
    response = client.get(
        "/get-age-by-photo",
        params={
            'url': 'https://indasil.club/uploads/posts/2022-11/1669494194_45-indasil-club-p-portret-starogo-cheloveka-instagram-51.jpg',
        }
    )
    assert response.status_code == 200
    assert response.json() == {
        'Age': 'more than 70'}


def test_young_man(client):
    response = client.get(
        "/get-age-by-photo",
        params={
            'url': 'https://luckclub.ru/images/luckclub/2019/03/newborn-baby-asleep-1.jpg',
        }
    )
    assert response.status_code == 200
    assert response.json() == {
        'Age': '0-2'}

def test_pustoi_query(client):
    response = client.get(
        "/get-age-by-photo",
        params={
            'url': 'абв',
        }
    )
    assert response.status_code == 200
    assert response.json() == {
        'Age': 'This is not url'}

# python -m pytest app/tests/test.py

